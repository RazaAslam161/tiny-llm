"""Autoregressive generation: a naive O(T^2) path and a KV-cached O(T) path.

Given identical logits the two produce identical tokens — that equivalence is
what tests/test_kv_cache.py pins down. The cache is a pure speed optimization;
it never changes what the model would have said.

Context ceiling: this model uses *learned absolute* position embeddings, so it
only knows positions 0..block_size-1. Generation is therefore capped at
block_size total tokens. (A sliding window past that would re-index positions,
which a KV cache — with positions baked into stored K/V — cannot replicate; it
is exactly why production models use relative positions like RoPE/ALiBi. Both
paths here cap identically so they stay interchangeable on every valid input.)

Batch: single-sequence only (B=1). A flat list of ids in, a flat list out.

Built with Claude Code.
"""

import torch

from serve.kv_cache import KVCache
from serve.sampler import sample


def _budget(model, ids, max_new_tokens):
    """Tokens we can actually generate before hitting the position ceiling."""
    if not isinstance(ids, (list, tuple)) or any(not isinstance(i, int) for i in ids):
        raise ValueError("ids must be a flat list of token ids (batch B=1 only)")
    block = model.config.block_size
    if len(ids) >= block:
        raise ValueError(f"prompt length {len(ids)} >= block_size {block}; no room to generate")
    return min(max_new_tokens, block - len(ids))


@torch.no_grad()
def generate_naive(model, ids, max_new_tokens, temperature=0.8, top_k=50, top_p=1.0):
    """Recompute attention over the whole prefix every step (no cache)."""
    model.eval()
    n = _budget(model, ids, max_new_tokens)
    idx = torch.tensor([ids], dtype=torch.long)
    for _ in range(n):
        logits, _ = model(idx)
        nxt = sample(logits[:, -1, :], temperature, top_k, top_p)
        idx = torch.cat((idx, nxt), dim=1)
    return idx[0].tolist()


@torch.no_grad()
def generate_cached(model, ids, max_new_tokens, temperature=0.8, top_k=50, top_p=1.0):
    """Prefill the prompt once, then feed one token at a time against the cache."""
    model.eval()
    n = _budget(model, ids, max_new_tokens)
    cache = KVCache()
    out = list(ids)
    cur = torch.tensor([ids], dtype=torch.long)  # prefill: whole prompt at once
    for _ in range(n):
        logits, _, presents = model(cur, past_kvs=cache.layers, use_cache=True)
        cache.update(presents)
        nxt = sample(logits[:, -1, :], temperature, top_k, top_p)
        out.append(int(nxt))
        cur = nxt  # thereafter: a single new token per step
    return out
