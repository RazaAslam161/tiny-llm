"""Correctness tests for KV-cached generation.

The cache is only allowed to make generation faster, never different. These
tests prove the cached path computes the same logits and emits the same tokens
as the naive recompute-everything path.
"""

import pytest
import torch

from model.config import GPTConfig
from model.transformer import GPT
from serve.engine import generate_cached, generate_naive
from serve.kv_cache import KVCache

SMALL = GPTConfig(vocab_size=512, block_size=64, n_layer=2, n_head=2, n_embd=64)


@pytest.fixture()
def model():
    torch.manual_seed(0)
    m = GPT(SMALL)
    m.eval()
    return m


def test_cached_logits_match_full_forward(model):
    """Feeding tokens one at a time through the cache must reproduce, at every
    position, the logits of a single full-sequence forward pass."""
    torch.manual_seed(1)
    T = 40
    seq = torch.randint(0, SMALL.vocab_size, (1, T))

    with torch.no_grad():
        full_logits, _ = model(seq)

        cache = KVCache()
        for t in range(T):
            step = seq[:, t:t + 1]
            logits, _, presents = model(step, past_kvs=cache.layers, use_cache=True)
            cache.update(presents)
            assert torch.allclose(logits[:, -1, :], full_logits[:, t, :], atol=1e-5), (
                f"cached logits diverge from full forward at position {t}"
            )
    assert cache.length == T


def test_greedy_generation_identical(model):
    prompt = [1, 5, 9, 2, 7]
    naive = generate_naive(model, prompt, max_new_tokens=30, temperature=0)
    cached = generate_cached(model, prompt, max_new_tokens=30, temperature=0)
    assert naive == cached


def test_sampled_generation_identical_with_same_seed(model):
    prompt = [3, 1, 4, 1, 5]
    torch.manual_seed(123)
    naive = generate_naive(model, prompt, max_new_tokens=25, temperature=0.9, top_k=40)
    torch.manual_seed(123)
    cached = generate_cached(model, prompt, max_new_tokens=25, temperature=0.9, top_k=40)
    assert naive == cached


def test_cache_grows_by_one_per_step(model):
    cache = KVCache()
    assert cache.length == 0
    with torch.no_grad():
        _, _, presents = model(torch.tensor([[1, 2, 3]]), use_cache=True)
        cache.update(presents)
        assert cache.length == 3  # prefill
        _, _, presents = model(torch.tensor([[4]]), past_kvs=cache.layers, use_cache=True)
        cache.update(presents)
        assert cache.length == 4  # one decode step


def test_multi_token_continuation_matches_full_forward(model):
    """Feeding a CHUNK of >1 new tokens against a non-empty cache must match the
    full forward — this exercises the mask slice for T>1 with past_len>0, which
    single-token decoding never hits."""
    torch.manual_seed(3)
    T, split = 24, 10
    seq = torch.randint(0, SMALL.vocab_size, (1, T))
    with torch.no_grad():
        full_logits, _ = model(seq)
        _, _, presents = model(seq[:, :split], use_cache=True)          # prefill 10
        cache = KVCache().update(presents)
        logits, _, _ = model(seq[:, split:], past_kvs=cache.layers, use_cache=True)  # 14 at once
        assert torch.allclose(logits[0], full_logits[0, split:], atol=1e-5)


def test_generation_caps_at_block_size_identically(model):
    """Both paths must stop at the position ceiling — same length, same tokens —
    rather than one crashing and one sliding."""
    prompt = [1, 2, 3, 4]
    huge = SMALL.block_size * 3
    naive = generate_naive(model, prompt, huge, temperature=0)
    cached = generate_cached(model, prompt, huge, temperature=0)
    assert len(naive) == SMALL.block_size
    assert naive == cached


def test_batch_input_rejected(model):
    with pytest.raises(ValueError):
        generate_naive(model, [[1, 2], [3, 4]], 5)
    with pytest.raises(ValueError):
        generate_cached(model, [[1, 2], [3, 4]], 5)
