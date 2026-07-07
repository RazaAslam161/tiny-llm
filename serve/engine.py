"""Autoregressive generation: a naive O(T^2) path and a KV-cached O(T) path.

The two must produce identical tokens given identical logits — that equivalence
is what tests/test_kv_cache.py pins down. The cache is a pure speed
optimization; it must never change what the model would have said.

Built with Claude Code.
"""

import torch
import torch.nn.functional as F

from serve.kv_cache import KVCache


def sample_next(logits, temperature=0.8, top_k=50):
    """Pick the next token id from last-position logits (B, vocab)."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_naive(model, ids, max_new_tokens, temperature=0.8, top_k=50):
    """Recompute attention over the whole prefix every step (no cache)."""
    model.eval()
    idx = torch.tensor([ids], dtype=torch.long)
    for _ in range(max_new_tokens):
        window = idx[:, -model.config.block_size:]
        logits, _ = model(window)
        nxt = sample_next(logits[:, -1, :], temperature, top_k)
        idx = torch.cat((idx, nxt), dim=1)
    return idx[0].tolist()


@torch.no_grad()
def generate_cached(model, ids, max_new_tokens, temperature=0.8, top_k=50):
    """Prefill the prompt once, then feed one token at a time against the cache."""
    model.eval()
    cache = KVCache()
    out = list(ids)
    cur = torch.tensor([ids], dtype=torch.long)  # prefill: whole prompt at once
    for _ in range(max_new_tokens):
        logits, _, presents = model(cur, past_kvs=cache.layers, use_cache=True)
        cache.update(presents)
        nxt = sample_next(logits[:, -1, :], temperature, top_k)
        out.append(int(nxt))
        cur = nxt  # thereafter: a single new token per step
    return out
