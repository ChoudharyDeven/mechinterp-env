"""
NumpyTransformer — Pure numpy transformer engine for mechinterp-env.

Supports:
  - Full forward pass with activation caching
  - Head ablation (zero out a specific head's output)
  - Activation patching (replace activation at (layer, head, position))
  - Logit lens (project residual stream at any layer to token probs)
  - Attention pattern extraction
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AttentionHead:
    W_Q: np.ndarray   # (d_model, d_head)
    W_K: np.ndarray   # (d_model, d_head)
    W_V: np.ndarray   # (d_model, d_head)
    W_O: np.ndarray   # (d_head, d_model)


@dataclass
class TransformerLayer:
    heads: List[AttentionHead]
    W_mlp_in:  np.ndarray   # (d_model, d_mlp)
    W_mlp_out: np.ndarray   # (d_mlp, d_model)


@dataclass
class NumpyTransformer:
    W_E:     np.ndarray            # (vocab_size, d_model) — token embedding
    W_U:     np.ndarray            # (d_model, vocab_size) — unembedding
    layers:  List[TransformerLayer]
    d_model: int
    n_heads: int
    d_head:  int
    vocab_size: int


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-10)


def causal_mask(seq_len: int) -> np.ndarray:
    """Upper-triangular mask filled with -1e9 (blocks future attention)."""
    return np.triu(np.full((seq_len, seq_len), -1e9), k=1)


# ─────────────────────────────────────────────────────────────────────────────
# Forward pass
# ─────────────────────────────────────────────────────────────────────────────

def forward(
    model: NumpyTransformer,
    tokens: List[int],
    ablated_heads: Optional[Set[Tuple[int, int]]] = None,
    patch_map: Optional[Dict[Tuple[int, int, int], np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Run a full forward pass, returning a rich activation cache.

    Args:
        model:         NumpyTransformer instance
        tokens:        List of token ids
        ablated_heads: Set of (layer, head) tuples to zero out
        patch_map:     Dict mapping (layer, head, position) → replacement tensor

    Returns:
        cache dict with keys:
          'embed'  : (seq, d_model)
          'layers' : list of per-layer dicts
          'logits' : (seq, vocab_size)
          'probs'  : (seq, vocab_size)
    """
    ablated_heads = ablated_heads or set()
    patch_map     = patch_map or {}

    seq_len = len(tokens)
    x       = model.W_E[tokens].copy()   # (seq, d_model)
    mask    = causal_mask(seq_len)

    cache = {
        "embed":  x.copy(),
        "layers": [],
    }

    for l, layer in enumerate(model.layers):
        layer_cache = {
            "attn_out":      [],   # list of (seq, d_model) per head
            "attn_patterns": [],   # list of (seq, seq) per head, or None
            "pre_mlp":       None,
            "post_layer":    None,
        }

        attn_sum = np.zeros_like(x)

        for h, head in enumerate(layer.heads):
            ablated = (l, h) in ablated_heads

            if ablated:
                head_out = np.zeros_like(x)
                layer_cache["attn_patterns"].append(None)
            else:
                Q = x @ head.W_Q                        # (seq, d_head)
                K = x @ head.W_K                        # (seq, d_head)
                V = x @ head.W_V                        # (seq, d_head)

                scale   = np.sqrt(model.d_head)
                scores  = Q @ K.T / scale + mask        # (seq, seq)
                attn    = softmax(scores, axis=-1)      # (seq, seq)
                context = attn @ V                      # (seq, d_head)
                head_out = context @ head.W_O            # (seq, d_model)

                layer_cache["attn_patterns"].append(attn.tolist())

            # Apply activation patches
            for pos in range(seq_len):
                key = (l, h, pos)
                if key in patch_map:
                    head_out[pos] = patch_map[key]

            layer_cache["attn_out"].append(head_out.tolist())
            attn_sum += head_out

        # Residual addition after attention
        x = x + attn_sum
        layer_cache["pre_mlp"] = x.copy().tolist()

        # MLP (ReLU activation)
        h_mlp = np.maximum(0, x @ layer.W_mlp_in)   # (seq, d_mlp)
        x     = x + h_mlp @ layer.W_mlp_out

        layer_cache["post_layer"] = x.copy().tolist()
        cache["layers"].append(layer_cache)

    logits = x @ model.W_U                            # (seq, vocab_size)
    probs  = softmax(logits, axis=-1)

    cache["logits"] = logits.tolist()
    cache["probs"]  = probs.tolist()

    return cache


# ─────────────────────────────────────────────────────────────────────────────
# Logit lens
# ─────────────────────────────────────────────────────────────────────────────

def logit_lens(
    model: NumpyTransformer,
    tokens: List[int],
    layer: int,
    position: int,
) -> np.ndarray:
    """
    Project the residual stream at a given layer and position through
    the unembedding matrix. Returns a probability distribution over vocab.

    layer=0 means after embedding (before any transformer layer).
    layer=L means after layer L-1's output.
    """
    cache = forward(model, tokens)

    if layer == 0:
        residual = cache["embed"][position]
    else:
        residual = np.array(cache["layers"][layer - 1]["post_layer"])[position]

    logits = residual @ model.W_U
    return softmax(logits)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: get target token probability from cache
# ─────────────────────────────────────────────────────────────────────────────

def get_token_prob(cache: Dict, position: int, token_id: int) -> float:
    """Get the probability of a specific token at a specific position."""
    return float(np.array(cache["probs"])[position][token_id])
