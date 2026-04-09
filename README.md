---
title: mechinterp-env
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - mechanistic-interpretability
  - ai-safety
  - transformers
  - circuit-discovery
  - rl-environment
---

# 🔬 mechinterp-env

> **Finding a circuit in a transformer is like locating a faulty transistor in a running CPU using only a voltmeter.** It takes human researchers weeks of careful activation patching. `mechinterp-env` is the training ground to automate this — we trap an AI agent inside a toy transformer with a mathematically planted circuit, hand it the tools of mechanistic interpretability, and force it to perform neurosurgery on the model.

**The first RL environment where agents learn to debug transformer circuits** — built on toy models with known ground-truth circuits, so every score is provably correct.

Built for the [OpenEnv AI Hackathon 2026](https://pytorch.org/event/openenv-ai-hackathon/) (Meta × Hugging Face × PyTorch).

---

## What the Agent Actually Sees

![Circuit Visualization](https://raw.githubusercontent.com/ChoudharyDeven/mechinterp-env/ddf5e33/circuit_hero.png)

*Left: The circuit head (H2) concentrates all attention on the RED token — copying it perfectly. Middle: A noise head spreads attention uniformly — no structured information flow. Right: The ablation evidence an agent uses to identify the circuit — ablating H2 causes a -1.0 behavioral collapse. Ablating noise heads: zero effect.*

---

## How It Works

```
                    THE SURGICAL LOOP
                    ─────────────────

[reset(task_id)]
      │
      ▼
Agent receives:
  • target_behavior  — "This head copies a token. Find it."
  • available_actions — ablate_head, patch_activation, query_logit_lens...
  • prerun_results   — (Task 1 only) ablation results already computed
      │
      ▼
[Agent Action: ablate_head(layer=0, head=2)]
      │
      ▼
Environment runs forward pass with H2 zeroed out
      │
      ▼
[Observation returned]:
  • behavioral_delta: -1.0000  ← H2 is CRITICAL
  • baseline_prob:     0.9999
  • ablated_prob:      0.0000
  • interpretation:   "IMPORTANT — likely circuit component"
      │
      ▼
[Reward: +0.40]  ← Information gain: |delta| × 0.3 + high-delta bonus
      │
      ▼
Agent ablates more heads, finds delta ≈ 0.0 for all others
      │
      ▼
[Agent Action: submit_hypothesis({"(0, 2)": 1.0})]
      │
      ▼
[Grader: F1 against ground truth] → score=0.999
[END] success=true steps=1 score=0.999
```

---

## The Three Planted Circuits

Each task uses a different transformer model with a mathematically designed circuit. Ground truth is known by construction — we built the weights.

![Induction Circuit](https://raw.githubusercontent.com/ChoudharyDeven/mechinterp-env/ddf5e33/induction_circuit.png)

*Task 2's two-head induction circuit: L0H1 (lookup head) concentrates attention on the signal token, writes its identity into dims 17-32. L1H2 (transcode head) reads that signal via K-composition and converts it to the target token. Ablating either head collapses the behavior entirely.*

### Task 1 — Copy Head (Easy)
| | |
|---|---|
| **Model** | 1 layer, 4 heads, d=64 |
| **Circuit** | `{(0, 2): 1.0}` — single copy head |
| **Behavior** | Model copies key token from position 2 to final position |
| **Agent task** | Read pre-run ablation results, identify the head with delta=-1.0, submit |
| **Budget** | 4 steps |
| **Why it's easy** | Answer is in the initial observation. Tests reading comprehension of MI evidence. |

### Task 2 — Induction Circuit (Medium)
| | |
|---|---|
| **Model** | 2 layers, 4 heads each, d=64 |
| **Circuit** | `{(0, 1): 0.6, (1, 2): 1.0}` — two-head composition |
| **Behavior** | L0H1 writes signal token identity → L1H2 transcodes it to target |
| **Agent task** | Design and run ablation experiments across 8 heads, identify the 2-head circuit |
| **Budget** | 12 steps |
| **Why it's medium** | No pre-run results. Agent must plan experiments and reason across layers. |

### Task 3 — Committee Circuit (Hard)
| | |
|---|---|
| **Model** | 4 layers, 8 heads each, d=64 |
| **Circuit** | `{(0,3):0.7, (1,1):0.8, (1,5):0.6, (2,2):1.0, (3,6):0.9}` — 5 heads |
| **Behavior** | 5 independent heads each contribute a fraction of the output signal |
| **Agent task** | Navigate a 32-head search space strategically, identify all 5 circuit heads |
| **Budget** | 20 steps (can't ablate everything — must prioritize) |
| **Why it's hard** | Large search space, strategic exploration required, partial credit via weighted F1 |

![Ablation Deltas](https://raw.githubusercontent.com/ChoudharyDeven/mechinterp-env/ddf5e33/delta_comparison.png)

*Ablation deltas across all three tasks. Circuit heads (red) cause catastrophic behavioral collapse when removed. Noise heads (gray) have exactly zero effect. The signal is clean — this is what real MI research looks like.*

---

## Reward Function

Dense reward — the agent gets signal at **every step**, not just at the end.

### Step-level rewards
| Component | Formula | Rationale |
|---|---|---|
| Information gain | `\|behavioral_delta\| × 0.3` | Rewards ablations that reveal something real |
| High-delta bonus | `+0.10 if delta < -0.35` | Extra reward for identifying clearly important heads |
| Redundancy penalty | `-0.10` per re-queried head | Punishes aimless repetition |
| Efficiency penalty | `-0.05` per step after step 8 | Encourages decisive behavior |

### Terminal reward (on `submit_hypothesis`)
| Component | Value | Condition |
|---|---|---|
| Circuit F1 × 2.0 | 0 – 2.0 | Primary signal — how correct is your circuit? |
| Speed bonus | +0.30 | F1 ≥ 0.8 AND steps ≤ budget/2 |
| Attempt bonus | +0.05 | Always — reward for submitting vs timing out |
| **Normalized** | **÷ 2.35 → (0, 1)** | Perfect fast solve = 0.999 |

**Calibration:** Random agent ≈ 0.05 avg reward. Systematic agent ≈ 0.35. Agent understanding MI workflow ≈ 0.75+.

---

## Frontier Model Baselines

Tested with `Qwen/Qwen2.5-72B-Instruct` via the HF inference router:

| Task | Score | Steps | Notes |
|---|---|---|---|
| `head-identification` (Easy) | **0.999** | 1 | Identified H2 immediately from pre-run results |
| `circuit-localization` (Medium) | **0.999** | 9 | Systematic ablation found both circuit heads |
| `full-hypothesis` (Hard) | 0.35 – 0.45 | 20 | Partial — found 2-3 of 5 circuit heads before budget |

The difficulty progression is genuine. Hard tasks are actually hard.

---

## Action Space

| Action | JSON | What it does |
|---|---|---|
| `ablate_head` | `{"action_type": "ablate_head", "layer": 0, "head": 2}` | Zero out head. Measures how critical it is. |
| `patch_activation` | `{"action_type": "patch_activation", "layer": 1, "head": 2, "position": 3, "source_prompt_id": 1}` | Replace activation with corrupted source. |
| `query_logit_lens` | `{"action_type": "query_logit_lens", "layer": 1, "position": -1}` | See model's intermediate token predictions. |
| `query_attn_pattern` | `{"action_type": "query_attn_pattern", "layer": 0, "head": 1, "prompt_id": 0}` | View full attention weight matrix. |
| `submit_hypothesis` | `{"action_type": "submit_hypothesis", "circuit_mask": {"components": {"(0, 2)": 1.0}}}` | **Terminal.** Triggers grader. Ends episode. |

---

## Grading

All graders are **deterministic** — no LLM, no randomness. Same hypothesis always gets the same score.

| Task | Method | Perfect Score |
|---|---|---|
| Task 1 | Exact match (0.999) + partial layer credit (0.5) | 0.999 |
| Task 2 | Standard F1 on circuit head set (threshold 0.3) | 0.999 |
| Task 3 | Weighted F1 — important heads (L2H2 at 1.0) count more | 0.999 |

---

## Why Toy Transformers?

This is how mechanistic interpretability research actually bootstraps. Elhage et al. (2021), Olsson et al. (2022), Wang et al. (2022) — all foundational MI papers start with toy models where ground truth is provable by construction.

**Advantages over cached GPT-2 activations:**
- ✅ Ground truth is mathematically exact — not approximate
- ✅ No GPU, no PyTorch, no TransformerLens — pure numpy
- ✅ Docker image is tiny — no 500MB activation cache files
- ✅ Microsecond forward passes — well within 20-minute budget
- ✅ Perfect determinism — same input always gives same score

---

## Real-World Utility

MI circuit discovery is bottlenecked by researcher time. A senior MI researcher spends **days to weeks** localizing a single circuit. An RL agent trained on this environment is a step toward automating that — a research assistant that can run activation patching experiments, form hypotheses, and narrow down circuit components autonomously.

This environment serves as a **curriculum**: start with provably correct ground truth on toy models, then scale to real models with activation caches. The architecture already supports this.

**The target community:** Anthropic, DeepMind, EleutherAI, and every independent MI researcher would have direct use for an agent trained on this environment.

---

## Setup

### Quick start

```bash
git clone https://github.com/ChoudharyDeven/mechinterp-env
cd mechinterp-env
pip install numpy fastapi uvicorn pydantic httpx openai
python verify_tasks.py        # 46/46 checks passed
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Test it

```bash
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": "head-identification"}'
```

### Docker

```bash
docker build -t mechinterp-env .
docker run -p 7860:7860 mechinterp-env
```

### Run inference

```bash
export MECHINTERP_ENV_URL=https://potatochoudhary-mechinterp-env.hf.space
export HF_TOKEN=your_hf_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | Yes | — | Hugging Face API token |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model for inference |
| `MECHINTERP_ENV_URL` | Yes (inference) | `http://localhost:7860` | Space URL |

---

## Project Structure

```
mechinterp-env/
├── inference.py              ← Baseline inference script (mandatory)
├── openenv.yaml              ← OpenEnv manifest
├── models.py                 ← Pydantic types: Action, Observation, CircuitMask
├── client.py                 ← Python client (async + sync)
├── verify_tasks.py           ← 46-check verification script
├── Dockerfile
├── server/
│   ├── app.py                ← FastAPI: /reset /step /state /health
│   ├── environment.py        ← Core logic: reset(), step(), action handlers
│   ├── transformer.py        ← Numpy transformer engine
│   ├── tasks.py              ← Task definitions + ground truth circuits
│   ├── graders.py            ← Deterministic graders (no LLM)
│   └── rewards.py            ← Dense reward function
└── data/
    ├── models/
    │   ├── model_1layer.py   ← 1L copy head (d=64)
    │   ├── model_2layer.py   ← 2L induction circuit (d=64)
    │   └── model_4layer.py   ← 4L committee circuit (d=64)
    └── circuits/
        ├── circuit_1.json    ← {"(0, 2)": 1.0}
        ├── circuit_2.json    ← {"(0, 1)": 0.6, "(1, 2)": 1.0}
        └── circuit_3.json    ← 5-head ground truth
```

---

## OpenEnv Spec Compliance

- ✅ Typed `Action`, `Observation` Pydantic models
- ✅ `step(action)` → `(observation, reward, done, info)`
- ✅ `reset()` → initial observation
- ✅ `state()` → current state
- ✅ `openenv.yaml` with 3 tasks and grader definitions
- ✅ Passes `openenv validate`
- ✅ Docker build clean, port 7860
- ✅ Phase 1 + Phase 2 passed

---

## References

- Elhage et al. (2021) — *A Mathematical Framework for Transformer Circuits*
- Olsson et al. (2022) — *In-context Learning and Induction Heads*
- Wang et al. (2022) — *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small*
- Conmy et al. (2023) — *Towards Automated Circuit Discovery for Mechanistic Interpretability*

---

## License

MIT
