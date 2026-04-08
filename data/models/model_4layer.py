"""
Model 3 — 4-layer, 8-head transformer with a planted 5-head committee circuit.

Architecture: 4 layers, 8 heads each, d_model=64, d_head=8, vocab=30
Circuit (5 heads, one per committee member):
  L0H3 — subject reader (importance 0.7)
  L1H1 — semantic reader (importance 0.8)
  L1H5 — verb classifier  (importance 0.6)
  L2H2 — cross-composer   (importance 1.0)
  L3H6 — output writer    (importance 0.9)

Ground truth: {(0,3):0.7, (1,1):0.8, (1,5):0.6, (2,2):1.0, (3,6):0.9}

Prompt: [2, 6, 11, 2, 0]  (The chef cooked the [?])
Target: tok16 (food token) at final position (PAD, tok0).

Committee circuit design:
  Each of the 5 circuit heads independently contributes a fraction of the
  signal needed to predict tok16. Together they sum to enough. Ablating
  any single head removes ~20% of total signal, dropping the logit below
  the competing tok0 self-signal — causing a visible behavioral delta.

  Key parameters:
    W_U[40, 16] = 50.0   — output dim 40 strongly predicts tok16
    W_E[0, 0]   = 10000  — PAD has a huge self-signal in dim 0
    Each circuit head contributes c=7 to residual dim 40 via W_V and W_O.
    5 heads: 5*7*50 = 1750 logit → tok16 wins (> 10000*8=80000? No...)

  Actually the logit math:
    residual[40] = 5 * attn * W_V * W_O ≈ 5 * 1 * 7 * 7 * (SIG^2/scale) = ~1960
    logit[16] = residual[40] * W_U[40,16] = 1960 * 50 = 98000
    logit[0]  = W_E[0,0] * W_U[0,0] = 10000 * 8 = 80000
    Baseline: 98000 > 80000 → tok16 wins ✓
    Ablate one head: residual[40] ≈ 1568, logit[16] ≈ 78400 < 80000 → tok0 wins ✓
    Noise heads: W_O ≈ 1e-4 → negligible residual contribution → no delta ✓
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from server.transformer import NumpyTransformer, TransformerLayer, AttentionHead


def build_model_4layer(seed: int = 7) -> NumpyTransformer:
    rng = np.random.default_rng(seed)

    VOCAB  = 30
    D      = 64
    D_H    = 8
    N_H    = 8
    D_MLP  = 128
    SIG    = 15.0
    ID     = 8.0
    c      = 7.0      # per-head contribution weight

    # ── Token embeddings ──────────────────────────────────────────────────
    W_E = np.zeros((VOCAB, D))
    for t in range(VOCAB):
        W_E[t, t % D] = ID
    W_E += rng.normal(0, 0.002, (VOCAB, D))

    # Subject token (tok6) broadcasts key signal in dim 51
    W_E[6,  51] = SIG
    # Target token (tok16) has key in dim 50 (not used for attention here, for future use)
    W_E[16, 50] = SIG

    # PAD (tok0) has 5 query signals (one per circuit head) in dims 13-17
    for dim in range(13, 18):
        W_E[0, dim] = SIG

    # PAD self-signal in dim 0: must dominate when any single circuit head is ablated.
    # W_E[0,0]=10000 → tok0 logit ≈ 80000.
    # 4 circuit heads → residual[40] ≈ 1568 → logit[16] ≈ 78400 < 80000 → tok0 wins.
    W_E[0, 0] = 10000.0

    # ── Unembedding ───────────────────────────────────────────────────────
    W_U = np.zeros((D, VOCAB))
    for t in range(VOCAB):
        W_U[t % D, t] = ID
    W_U += rng.normal(0, 0.002, (D, VOCAB))

    # Output subspace: dim 40 strongly predicts tok16
    W_U[40, 16] = 50.0

    # ── Circuit head definitions ──────────────────────────────────────────
    # Each circuit head: PAD queries dim (13+i), subject (tok6) keys dim 51,
    # value reads tok6 identity (dim 6), output writes to residual dim 40.
    CIRCUIT_SPECS = [
        (0, 3, 13),   # layer 0, head 3, query_dim 13
        (1, 1, 14),   # layer 1, head 1, query_dim 14
        (1, 5, 15),   # layer 1, head 5, query_dim 15
        (2, 2, 16),   # layer 2, head 2, query_dim 16
        (3, 6, 17),   # layer 3, head 6, query_dim 17
    ]

    # ── Build all layers ──────────────────────────────────────────────────
    def noise_head():
        return AttentionHead(
            W_Q = rng.normal(0, 1e-4, (D, D_H)),
            W_K = rng.normal(0, 1e-4, (D, D_H)),
            W_V = rng.normal(0, 1e-4, (D, D_H)),
            W_O = rng.normal(0, 1e-4, (D_H, D)),
        )

    all_layers = []
    for layer_idx in range(4):
        heads = [noise_head() for _ in range(N_H)]

        for (l, hpos, q_dim) in CIRCUIT_SPECS:
            if l != layer_idx:
                continue

            W_Q = np.zeros((D, D_H))
            W_K = np.zeros((D, D_H))
            W_V = np.zeros((D, D_H))
            W_O = np.zeros((D_H, D))

            W_Q[q_dim, 0] = SIG     # PAD queries via its unique query dim
            W_K[51,    0] = SIG     # tok6 (subject) keys via dim 51
            W_V[6,     0] = c       # reads tok6 identity (dim 6)
            W_O[0,    40] = c       # writes to output subspace dim 40

            # Small noise to avoid perfect symmetry
            W_Q += rng.normal(0, 1e-4, (D, D_H))
            W_K += rng.normal(0, 1e-4, (D, D_H))
            W_O += rng.normal(0, 1e-4, (D_H, D))

            heads[hpos] = AttentionHead(W_Q=W_Q, W_K=W_K, W_V=W_V, W_O=W_O)

        all_layers.append(TransformerLayer(
            heads     = heads,
            W_mlp_in  = rng.normal(0, 1e-4, (D, D_MLP)),
            W_mlp_out = rng.normal(0, 1e-4, (D_MLP, D)),
        ))

    return NumpyTransformer(
        W_E=W_E, W_U=W_U,
        layers=all_layers,
        d_model=D, n_heads=N_H, d_head=D_H, vocab_size=VOCAB,
    )


if __name__ == "__main__":
    from server.transformer import forward, get_token_prob
    model  = build_model_4layer()
    tokens = [2, 6, 11, 2, 0]
    TARGET = 16
    cache  = forward(model, tokens)
    base   = get_token_prob(cache, -1, TARGET)
    print(f"Baseline prob for tok16: {base:.4f}")

    CIRCUIT = [(0,3),(1,1),(1,5),(2,2),(3,6)]
    print("Circuit heads:")
    for (l, h) in CIRCUIT:
        ac    = forward(model, tokens, ablated_heads={(l, h)})
        p     = get_token_prob(ac, -1, TARGET)
        delta = p - base
        print(f"  L{l}H{h}: delta={delta:.4f}  {'*** CIRCUIT' if abs(delta) > 0.1 else ''}")

    print("Noise heads (sample):")
    for (l, h) in [(0,0),(0,1),(1,0),(2,0),(3,0)]:
        ac    = forward(model, tokens, ablated_heads={(l, h)})
        p     = get_token_prob(ac, -1, TARGET)
        delta = p - base
        print(f"  L{l}H{h}: delta={delta:.4f}")
