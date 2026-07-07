"""Two generation paths: naive recompute and kv-cached. Same tokens out of
either one, the cache is just speed. Capped at block_size since we use absolute
position embeddings and a sliding cache can't reindex the stored k/v. B=1 only.
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
    return list(ids) + list(
        stream_tokens(model, ids, max_new_tokens, temperature, top_k, top_p)
    )


@torch.no_grad()
def stream_tokens(model, ids, max_new_tokens, temperature=0.8, top_k=50, top_p=1.0):
    """Yield generated token ids one at a time (KV-cached) — for live streaming."""
    model.eval()
    n = _budget(model, ids, max_new_tokens)
    cache = KVCache()
    cur = torch.tensor([ids], dtype=torch.long)  # prefill: whole prompt at once
    for _ in range(n):
        logits, _, presents = model(cur, past_kvs=cache.layers, use_cache=True)
        cache.update(presents)
        nxt = sample(logits[:, -1, :], temperature, top_k, top_p)
        yield int(nxt)
        cur = nxt  # thereafter: a single new token per step
