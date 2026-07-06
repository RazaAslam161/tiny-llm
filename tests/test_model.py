"""Tests for the GPT decoder.

The causality test and the single-batch overfit catch most transformer
wiring bugs: a missing/shifted mask breaks the first, and broken gradients,
bad init, or mis-wired residuals break the second.
"""

import pytest
import torch

from model.config import GPTConfig
from model.transformer import GPT

SMALL = GPTConfig(vocab_size=512, block_size=64, n_layer=2, n_head=2, n_embd=64)


@pytest.fixture()
def small_model():
    torch.manual_seed(0)
    return GPT(SMALL)


def test_output_shape_and_loss(small_model):
    x = torch.randint(0, SMALL.vocab_size, (2, 16))
    y = torch.randint(0, SMALL.vocab_size, (2, 16))
    logits, loss = small_model(x, y)
    assert logits.shape == (2, 16, SMALL.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)
    logits, loss = small_model(x)
    assert logits.shape == (2, 16, SMALL.vocab_size)
    assert loss is None


def test_causality(small_model):
    """Changing the token at position t must not affect logits at positions < t."""
    small_model.eval()
    torch.manual_seed(1)
    x = torch.randint(0, SMALL.vocab_size, (1, 32))
    with torch.no_grad():
        base, _ = small_model(x)
        for t in [1, 5, 16, 31]:
            x2 = x.clone()
            x2[0, t] = (x2[0, t] + 1) % SMALL.vocab_size
            perturbed, _ = small_model(x2)
            assert torch.equal(base[:, :t], perturbed[:, :t]), (
                f"logits before position {t} changed — causal mask is broken"
            )
            assert not torch.equal(base[:, t], perturbed[:, t]), (
                f"logits at position {t} did not change — model ignores input?"
            )


def test_overfit_single_batch(small_model):
    """200 steps on one batch must memorize it. If not, the wiring is wrong."""
    torch.manual_seed(2)
    x = torch.randint(0, SMALL.vocab_size, (4, 32))
    y = torch.randint(0, SMALL.vocab_size, (4, 32))
    opt = torch.optim.AdamW(small_model.parameters(), lr=3e-3)
    small_model.train()
    loss = None
    for _ in range(200):
        _, loss = small_model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5, f"final loss {loss.item():.3f} — model failed to memorize one batch"


def test_param_count_is_about_12m():
    torch.manual_seed(0)
    model = GPT(GPTConfig())  # default: 6L/6H/384d, vocab 4096, ctx 256
    n = model.num_params()
    assert 11_000_000 < n < 13_500_000, f"param count {n:,} not ~12M"


def test_rejects_sequences_over_block_size(small_model):
    x = torch.randint(0, SMALL.vocab_size, (1, SMALL.block_size + 1))
    with pytest.raises(AssertionError):
        small_model(x)
