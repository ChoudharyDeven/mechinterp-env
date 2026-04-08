"""
Model 2 — 2-layer, 4-head transformer with a planted 2-head circuit.

Architecture: 2 layers, 4 heads each, d_model=64, d_head=16, vocab=12
Circuit:
  L0H1 — Lookup head:    PAD (pos7) attends to tok1 (pos6), writes tok1 identity to dims 17-32
  L1H2 — Transcode head: reads dims 17-32 (tok1 signal), outputs tok2 identity

Ground truth: {(0, 1): 0.6, (1, 2): 1.0}

Prompt: [3, 4, 5, 6, 7, 8, 1, 0]
  Positions 0-5: noise tokens (3,4,5,6,7,8) never appear at pos 6 or 7
  Position 6:    tok1  — the signal token (only occurrence in prompt)
  Position 7:    tok0  — PAD (the query position, predicts next token)
Target: tok2 at position 7.

How it works:
  L0H1: tok1 (pos6) has KEY signal in dim 12. PAD (pos7) has QUERY signal in dim 13.
         Q[dim13→hd0] · K[dim12→hd0] >> all other scores → PAD attends to tok1 only.
         V copies tok1 identity (dim 1 = ID) into head output dims 0-15.
         O writes to dims 17-32 of pos7's residual.
         Result: pos7 residual dims 17-32 contain tok1's identity.

  L1H2: Q reads dims 17-32 (= tok1 identity, written by L0H1).
         K reads current token identity dims 0-15 (tok1 at pos6 has dim 1 = ID).
         Score(pos7→pos6) >> all others → concentrated attention on tok1 position.
         V maps tok1 dim (dim 1) to head output dim 2 via W_V[1,2]=ID.
         O writes dim 2 to residual dim 2 (tok2's identity dimension).
         W_U[2,2]=ID → logit for tok2 dominates → predicts tok2. ✓

Why ablating L0H1 matters:
  Without L0H1, dims 17-32 at pos7 = 0. L1H2 Q = 0 → uniform attention.
  With uniform attention (1/8 weight on pos6), tok2 signal = 12.5 * ID = 125.
  W_E[0,0]=200 → tok0 logit from embedding = 200 * ID = 2000 >> 125 * ID = 1250.
  PAD (tok0) predicted instead of tok2 → behavioral_delta ≈ -1.0.

Why ablating L1H2 matters:
  Without L1H2, tok1 signal in dims 17-32 is never transcoded to tok2.
  No head produces tok2 dims in residual. Model predicts tok0 again. delta ≈ -1.0.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from server.transformer import NumpyTransformer, TransformerLayer, AttentionHead


def build_model_2layer(seed: int = 99) -> NumpyTransformer:
    rng = np.random.default_rng(seed)

    VOCAB = 12
    D     = 64
    D_H   = 16
    N_H   = 4
    D_MLP = 64
    SIG   = 15.0
    ID    = 10.0

    # ── Token embeddings: one-hot in dims 0-11 ────────────────────────────
    W_E = np.zeros((VOCAB, D))
    for t in range(VOCAB):
        W_E[t, t] = ID
    W_E += rng.normal(0, 0.002, (VOCAB, D))

    # L0H1 attention channel signals
    W_E[1, 12] = SIG    # tok1: KEY — broadcasts "I am the source"
    W_E[0, 13] = SIG    # PAD:  QUERY — "I want tok1's content"

    # PAD self-signal — dominates prediction when L0H1 is ablated.
    # Ablated L0H1 → uniform attn → tok2 signal = ID^3/8 ≈ 1250 (logit scale)
    # W_E[0,0]=200 → tok0 logit = 200*ID = 2000 >> 1250 → tok0 wins → large delta
    W_E[0, 0] = 200.0

    # ── Unembedding: one-hot recovery ─────────────────────────────────────
    W_U = np.zeros((D, VOCAB))
    for t in range(VOCAB):
        W_U[t, t] = ID
    W_U += rng.normal(0, 0.002, (D, VOCAB))

    # ── Noise heads ───────────────────────────────────────────────────────
    def noise_head():
        return AttentionHead(
            W_Q = rng.normal(0, 1e-4, (D, D_H)),
            W_K = rng.normal(0, 1e-4, (D, D_H)),
            W_V = rng.normal(0, 1e-4, (D, D_H)),
            W_O = rng.normal(0, 1e-4, (D_H, D)),
        )

    # ── L0H1: Lookup head ────────────────────────────────────────────────
    W_Q_L0H1 = np.zeros((D, D_H))
    W_K_L0H1 = np.zeros((D, D_H))
    W_V_L0H1 = np.zeros((D, D_H))
    W_O_L0H1 = np.zeros((D_H, D))

    W_Q_L0H1[13, 0] = SIG      # PAD query reads dim 13
    W_K_L0H1[12, 0] = SIG      # tok1 key reads dim 12

    for i in range(D_H):
        W_V_L0H1[i, i] = ID    # copy token identity dims 0..D_H-1
    for i in range(D_H):
        W_O_L0H1[i, 17 + i] = ID  # write to prev_token subspace dims 17-32

    L0H1 = AttentionHead(W_Q=W_Q_L0H1, W_K=W_K_L0H1, W_V=W_V_L0H1, W_O=W_O_L0H1)

    layer0 = TransformerLayer(
        heads     = [noise_head(), L0H1, noise_head(), noise_head()],
        W_mlp_in  = rng.normal(0, 1e-4, (D, D_MLP)),
        W_mlp_out = rng.normal(0, 1e-4, (D_MLP, D)),
    )

    # ── L1H2: Transcode head ─────────────────────────────────────────────
    W_Q_L1H2 = np.zeros((D, D_H))
    W_K_L1H2 = np.zeros((D, D_H))
    W_V_L1H2 = np.zeros((D, D_H))
    W_O_L1H2 = np.zeros((D_H, D))

    for i in range(D_H):
        W_Q_L1H2[17 + i, i] = SIG  # Q reads prev_token subspace dims 17-32
    for i in range(D_H):
        W_K_L1H2[i, i] = SIG       # K reads current token identity dims 0-15

    W_V_L1H2[1, 2] = ID            # tok1 dim 1 → head dim 2 (transcode)
    W_O_L1H2[2, 2] = ID            # head dim 2 → residual dim 2 (tok2 identity)

    L1H2 = AttentionHead(W_Q=W_Q_L1H2, W_K=W_K_L1H2, W_V=W_V_L1H2, W_O=W_O_L1H2)

    layer1 = TransformerLayer(
        heads     = [noise_head(), noise_head(), L1H2, noise_head()],
        W_mlp_in  = rng.normal(0, 1e-4, (D, D_MLP)),
        W_mlp_out = rng.normal(0, 1e-4, (D_MLP, D)),
    )

    return NumpyTransformer(
        W_E=W_E, W_U=W_U,
        layers=[layer0, layer1],
        d_model=D, n_heads=N_H, d_head=D_H, vocab_size=VOCAB,
    )


if __name__ == "__main__":
    from server.transformer import forward, get_token_prob
    model  = build_model_2layer()
    tokens = [3, 4, 5, 6, 7, 8, 1, 0]
    TARGET = 2
    cache  = forward(model, tokens)
    base   = get_token_prob(cache, -1, TARGET)
    print(f"Baseline prob for tok2: {base:.4f}")
    for l in range(2):
        for h in range(4):
            ac    = forward(model, tokens, ablated_heads={(l, h)})
            p     = get_token_prob(ac, -1, TARGET)
            delta = p - base
            mark  = " *** CIRCUIT" if abs(delta) > 0.2 else ""
            print(f"  L{l}H{h}: delta={delta:.4f}{mark}")
