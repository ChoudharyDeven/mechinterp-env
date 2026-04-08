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

# mechinterp-env

**The first RL environment where agents learn to debug transformer circuits — built on toy models with known ground-truth circuits, so every score is provably correct.**

Built for the [OpenEnv AI Hackathon 2026](https://pytorch.org/event/openenv-ai-hackathon/) (Meta × Hugging Face × PyTorch).

---

## What This Is

`mechinterp-env` is a reinforcement learning environment for **mechanistic interpretability (MI) circuit debugging**. An agent is placed inside a world containing a small transformer model with a known behavioral anomaly. The agent's job is to identify which attention heads are responsible — the "circuit" — using the same tools MI researchers use:

- **Activation patching** — replace an activation and measure behavioral change
- **Head ablation** — zero out a head and see how much the behavior drops
- **Logit lens** — inspect the model's intermediate token predictions
- **Attention pattern queries** — see which tokens a head attends to

The agent submits a **circuit hypothesis** (a mapping of heads to importance scores). The grader compares it to the known ground-truth circuit and returns a score in [0, 1].

---

## Why Toy Transformers?

`mechinterp-env` uses hand-crafted numpy transformers with **planted circuits** rather than pre-trained models. This is the correct approach for RL training because:

1. **Ground truth by construction** — we designed the weights, so the circuit is known exactly. Graders are provably correct.
2. **Scientific legitimacy** — this is how foundational MI research works (see Elhage et al., Olsson et al.). Toy models with known circuits are the standard starting point.
3. **No GPU required** — pure numpy, microsecond forward passes, tiny Docker image.
4. **Perfect determinism** — same input always gives same output, same score.

---

## The Three Tasks

| Task | Model | Budget | Difficulty | What the agent must do |
|------|-------|--------|------------|------------------------|
| `head-identification` | 1L, 4H, d=32 | 4 steps | Easy | Read pre-run ablation results and identify the single copy head |
| `circuit-localization` | 2L, 4H, d=48 | 12 steps | Medium | Design experiments to discover a 2-head induction circuit |
| `full-hypothesis` | 4L, 8H, d=64 | 20 steps | Hard | Navigate a 32-head search space to find a 5-head multi-layer circuit |

### Planted Circuits

| Task | Ground Truth Circuit |
|------|---------------------|
| `head-identification` | `{(0, 2): 1.0}` — single copy head |
| `circuit-localization` | `{(0, 1): 0.6, (1, 2): 1.0}` — induction circuit (prev-token head + induction head) |
| `full-hypothesis` | `{(0,3):0.7, (1,1):0.8, (1,5):0.6, (2,2):1.0, (3,6):0.9}` — subject-verb-object circuit |

---

## Action Space

```json
{"action_type": "ablate_head", "layer": 0, "head": 2}
{"action_type": "patch_activation", "layer": 1, "head": 2, "position": 3, "source_prompt_id": 1}
{"action_type": "query_logit_lens", "layer": 1, "position": -1}
{"action_type": "query_attn_pattern", "layer": 0, "head": 1, "prompt_id": 0}
{"action_type": "submit_hypothesis", "circuit_mask": {"components": {"(0, 2)": 1.0}}}
```

## Observation Space

Each step returns a structured observation with:
- `target_behavior` — natural language description of the anomaly
- `action_result` — numerical result of the last action (deltas, probs, patterns)
- `behavioral_delta` — change in target behavior metric after the last action
- `experiment_history` — last 8 experiment records
- `steps_remaining` — budget remaining
- `available_actions` — valid actions for this task
- `done`, `reward`

---

## Reward Function

**Step-level (fires every action):**
- Information gain: `|behavioral_delta| × 0.3` — rewards informative experiments
- High-delta bonus: `+0.10` if `behavioral_delta < -0.35`
- Redundancy penalty: `−0.10` for re-querying the same head
- Efficiency penalty: `−0.05` per step after step 8

**Terminal (fires on `submit_hypothesis`):**
- Circuit F1 × 2.0 (primary signal)
- Speed bonus: +0.30 if F1 ≥ 0.8 and steps ≤ budget/2
- Attempt bonus: +0.05

Normalized to [0, 1]. Perfect fast solve = 1.0.

---

## Baseline Scores

| Task | Expected Score | Notes |
|------|---------------|-------|
| `head-identification` | 0.85 – 1.00 | Answer is in initial observation |
| `circuit-localization` | 0.50 – 0.75 | Requires systematic ablation |
| `full-hypothesis` | 0.25 – 0.45 | Hard — strategic search required |

---

## Setup & Usage

### Running locally

```bash
# Clone the repo
git clone https://huggingface.co/spaces/your-username/mechinterp-env
cd mechinterp-env

# Install dependencies
pip install numpy fastapi uvicorn pydantic httpx openai

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# In another terminal, run the baseline agent
MECHINTERP_ENV_URL=http://localhost:7860 \
HF_TOKEN=your_token \
python inference.py
```

### Docker

```bash
docker build -t mechinterp-env .
docker run -p 7860:7860 mechinterp-env

# Test it
curl -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": "head-identification"}'
```

### API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode. Body: `{"task_id": "..."}` |
| `/step` | POST | Take action. Body: action JSON object |
| `/state` | GET | Current episode state |
| `/health` | GET | Health check |

---

## Project Structure

```
mechinterp-env/
├── inference.py          # Baseline inference script (mandatory)
├── openenv.yaml          # OpenEnv manifest
├── models.py             # Pydantic types: Action, Observation, CircuitMask
├── client.py             # Python client (async + sync)
├── Dockerfile
├── pyproject.toml
├── verify_tasks.py       # Verification script — run before deploying
├── data/
│   ├── models/
│   │   ├── model_1layer.py   # 1L copy head model
│   │   ├── model_2layer.py   # 2L induction circuit model
│   │   └── model_4layer.py   # 4L multi-head circuit model
│   └── circuits/
│       ├── circuit_1.json    # Ground truth for task 1
│       ├── circuit_2.json    # Ground truth for task 2
│       └── circuit_3.json    # Ground truth for task 3
└── server/
    ├── app.py            # FastAPI app
    ├── environment.py    # Core environment logic
    ├── transformer.py    # Numpy transformer engine
    ├── tasks.py          # Task definitions
    ├── graders.py        # Deterministic graders
    ├── rewards.py        # Dense reward function
    └── requirements.txt
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes (inference) | — | Your Hugging Face API token |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model for inference |
| `MECHINTERP_ENV_URL` | Yes (inference) | `http://localhost:7860` | Space URL |

---

## Citation & Background

This environment draws on the following MI research:

- Elhage et al. (2021) — *A Mathematical Framework for Transformer Circuits*
- Olsson et al. (2022) — *In-context Learning and Induction Heads*
- Wang et al. (2022) — *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small*

The planted circuit methodology (constructing toy models with known circuits) is the standard approach in foundational MI research for establishing ground truth.

---

## License

MIT
