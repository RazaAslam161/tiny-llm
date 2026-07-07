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
