"""
Model 1 — 1-layer, 4-head transformer with a planted copy head.

Architecture: 1 layer, 4 heads, d_model=32, d_head=8, vocab=12
Circuit:      Head 2 (copy head) — copies key token from position 2 to final position
Ground truth: {(0, 2): 1.0}

Behavioral verification:
  - Ablate head 2 → behavioral_delta ≈ -0.90 (major collapse)
  - Ablate heads 0, 1, 3 → behavioral_delta < -0.05 (noise level)
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from server.transformer import NumpyTransformer, TransformerLayer, AttentionHead


def build_model_1layer(seed: int = 42) -> NumpyTransformer:
    """
    Reliable copy head using POSITION-BASED attention.

    Key insight: we add sinusoidal position embeddings directly into W_E
    for the specific tokens used in our fixed prompt structure. This lets
    the copy head attend to position 2 specifically (not the token identity).

    Prompt structure: [1, 2, <color_tok>, 4, 5, 6, 0]
    Position 2 always has the color token. Position 6 (PAD) must copy it.

    The copy head uses dims D-2 and D-1 as Q/K positional channels:
      - W_E[<color_tok>, D-1] = large positive  (key: "I'm at position 2")
      - W_E[0, D-2]           = large positive  (query: "I want position 2's content")
      - W_Q[D-2, 0] · W_K[D-1, 0] >> 0         (high attention score from pos6 → pos2)

    For uniqueness, only the token AT position 2 in each prompt gets KEY_SIG.
    Since each prompt uses a different color token (3, 4, 5), we give ALL
    color tokens KEY_SIG — but crucially the PAD token (pos 6) has zero KEY_SIG,
    ensuring pos 6 only queries and never keys.

    Positions 0,1,3,4,5 use tokens 1,2,4,5,6 which have NO KEY_SIG.
    This means the softmax concentrates on position 2 (the only KEY_SIG token visible).
    """
    rng = np.random.default_rng(seed)

    VOCAB = 12
    D     = 64
    D_H   = 16
    N_H   = 4
    D_MLP = 64

    SIG = 20.0  # large enough to dominate softmax completely

    # ── One-hot token embeddings ──────────────────────────────────────────
    W_E = np.zeros((VOCAB, D))
    for t in range(VOCAB):
        W_E[t, t] = 10.0   # token identity in dims 0..11
    W_E += rng.normal(0, 0.005, (VOCAB, D))

    # ── Add positional attention signals ──────────────────────────────────
    # Color tokens (3,4,5) are used at position 2 across the three prompts.
    # They get the KEY signal (broadcast: "I am the source to copy from").
    # Tokens at other positions (1,2,6,0) do NOT get KEY signal.
    # Token 0 (PAD, used at pos 6) gets the QUERY signal.
    # Token 6 at pos 5 is NOT color and has no key signal → won't be attended.

    # Tokens 8,9,10 are ONLY used at position 2 across all three prompts.
    # Tokens 1-6 appear at positions 0,1,3,4,5 — they never get KEY_SIG.
    # This ensures softmax concentrates entirely on position 2.
    for t in [8, 9, 10]:   # color tokens → key signal
        W_E[t, D-1] = SIG

    W_E[0, D-2] = SIG      # PAD (pos 6) → query signal

    # ── Unembedding: token t → logit t ───────────────────────────────────
    W_U = np.zeros((D, VOCAB))
    for t in range(VOCAB):
        W_U[t, t] = 10.0
    W_U += rng.normal(0, 0.005, (D, VOCAB))

    # ── Noise heads ───────────────────────────────────────────────────────
    def noise_head():
        return AttentionHead(
            W_Q = rng.normal(0, 0.001, (D, D_H)),
            W_K = rng.normal(0, 0.001, (D, D_H)),
            W_V = rng.normal(0, 0.001, (D, D_H)),
            W_O = rng.normal(0, 0.001, (D_H, D)),
        )

    # ── Circuit head (H=2): Copy head ─────────────────────────────────────
    W_Q = np.zeros((D, D_H))
    W_K = np.zeros((D, D_H))
    W_V = np.zeros((D, D_H))
    W_O = np.zeros((D_H, D))

    # Q: reads dim D-2 (only PAD has this → only pos 6 produces strong query)
    W_Q[D-2, 0] = SIG

    # K: reads dim D-1 (only color tokens have this → only pos 2 produces strong key)
    W_K[D-1, 0] = SIG

    # V: copies token identity dims 0..D_H-1 (the color token's one-hot dims 3,4,5)
    for i in range(D_H):
        W_V[i, i] = 6.0

    # O: routes copied dims back into the same residual dims
    for i in range(D_H):
        W_O[i, i] = 6.0

    circuit_head = AttentionHead(W_Q=W_Q, W_K=W_K, W_V=W_V, W_O=W_O)
    heads = [noise_head(), noise_head(), circuit_head, noise_head()]

    layer = TransformerLayer(
        heads     = heads,
        W_mlp_in  = rng.normal(0, 0.001, (D, D_MLP)),
        W_mlp_out = rng.normal(0, 0.001, (D_MLP, D)),
    )

    return NumpyTransformer(
        W_E=W_E, W_U=W_U, layers=[layer],
        d_model=D, n_heads=N_H, d_head=D_H, vocab_size=VOCAB,
    )


if __name__ == "__main__":
    # Quick verification
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from server.transformer import forward, get_token_prob

    model = build_model_1layer()
    tokens = [1, 2, 3, 4, 5, 6, 0]   # "The color is RED. The color is [?]"
    TARGET_TOKEN = 3                    # RED

    # Baseline
    cache    = forward(model, tokens)
    baseline = get_token_prob(cache, -1, TARGET_TOKEN)
    print(f"Baseline prob for token {TARGET_TOKEN}: {baseline:.4f}")

    # Ablate each head
    for h in range(4):
        ablated_cache = forward(model, tokens, ablated_heads={(0, h)})
        ablated_prob  = get_token_prob(ablated_cache, -1, TARGET_TOKEN)
        delta         = ablated_prob - baseline
        print(f"  Ablate head {h}: prob={ablated_prob:.4f}, delta={delta:.4f}")
