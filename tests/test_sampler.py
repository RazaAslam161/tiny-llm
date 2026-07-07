"""Tests for the sampler: the roadmap's required properties plus edge cases.

top_p=1.0 == plain sampling; p->0 == greedy; temperature 0 == argmax; top_k=1
== greedy; and top_p actually restricts the support to the nucleus.
"""

import torch

from serve.sampler import sample


def test_temperature_zero_is_argmax():
    logits = torch.tensor([[0.1, 3.0, 0.2, 2.9]])
    for _ in range(10):
        assert sample(logits, temperature=0).item() == 1


def test_top_k_1_is_greedy():
    torch.manual_seed(0)
    logits = torch.randn(1, 64)
    top = logits.argmax(-1).item()
    for _ in range(20):
        assert sample(logits, temperature=1.0, top_k=1).item() == top


def test_top_p_near_zero_is_greedy():
    torch.manual_seed(1)
    logits = torch.randn(1, 64)
    top = logits.argmax(-1).item()
    for _ in range(20):
        assert sample(logits, temperature=1.0, top_p=1e-6).item() == top


def test_top_p_one_equals_plain_sampling():
    # with the same generator seed, top_p=1.0 (no filtering) draws exactly what a
    # bare softmax+multinomial would
    torch.manual_seed(2)
    logits = torch.randn(1, 100)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = sample(logits, temperature=1.0, top_p=1.0, generator=g1)
    b = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=g2)
    assert a.item() == b.item()


def test_top_p_restricts_to_nucleus():
    # peaked distribution: the deep tail must never be sampled under top_p=0.9
    logits = torch.tensor([[5.0, 4.0, 0.0, -5.0, -10.0]])
    counts = torch.zeros(5)
    g = torch.Generator().manual_seed(3)
    for _ in range(3000):
        counts[sample(logits, temperature=1.0, top_p=0.9, generator=g).item()] += 1
    assert counts[3] == 0 and counts[4] == 0


def test_batched_logits():
    logits = torch.tensor([[0.0, 9.0], [9.0, 0.0]])
    out = sample(logits, temperature=0)
    assert out.shape == (2, 1)
    assert out[0].item() == 1 and out[1].item() == 0
