"""Tests for int8 per-channel weight quantization."""

import copy

import torch
import torch.nn as nn

from model.config import GPTConfig
from model.transformer import GPT
from serve.quantize import (
    QuantizedLinear,
    quantize_gpt,
    quantize_model_,
    quantize_weight,
)

SMALL = GPTConfig(vocab_size=512, block_size=64, n_layer=2, n_head=2, n_embd=64)


def test_quantize_weight_error_within_one_step():
    torch.manual_seed(0)
    w = torch.randn(16, 32)
    q, scale = quantize_weight(w)
    assert q.dtype == torch.int8
    deq = q.float() * scale[:, None]
    # each element is within half a quantization step of the original
    assert (deq - w).abs().max().item() <= scale.max().item()


def test_quantized_linear_matches_fp32():
    torch.manual_seed(1)
    lin = nn.Linear(64, 128, bias=False)
    ql = QuantizedLinear.from_linear(lin)
    x = torch.randn(4, 64)
    rel = (ql(x) - lin(x)).norm() / lin(x).norm()
    assert rel < 0.02


def test_per_channel_preserves_small_magnitude_row():
    # a row 1000x smaller than another: a single per-tensor scale would crush it,
    # per-channel keeps its resolution
    torch.manual_seed(2)
    w = torch.stack([torch.randn(128) * 1.0, torch.randn(128) * 1e-3])
    q, scale = quantize_weight(w)
    deq = q.float() * scale[:, None]
    rel_small = (deq[1] - w[1]).norm() / w[1].norm()
    assert rel_small < 0.02


def test_quantize_model_preserves_shape_and_close_logits():
    torch.manual_seed(0)
    model = GPT(SMALL)
    model.eval()
    x = torch.randint(0, SMALL.vocab_size, (1, 16))
    y0, _ = model(x)

    qm = quantize_model_(copy.deepcopy(model))
    qm.eval()
    y1, _ = qm(x)

    assert y1.shape == y0.shape
    assert (y1 - y0).norm() / y0.norm() < 0.05
    qlinears = [m for m in qm.modules() if isinstance(m, QuantizedLinear)]
    assert qlinears
    assert all(m.qweight.dtype == torch.int8 for m in qlinears)


def test_quantized_weights_are_one_byte():
    torch.manual_seed(0)
    lin = nn.Linear(100, 200, bias=False)
    ql = QuantizedLinear.from_linear(lin)
    assert ql.qweight.element_size() == 1
    assert ql.scale.numel() == 200  # one scale per output channel


def test_quantize_gpt_shares_tied_embedding_and_stays_close():
    torch.manual_seed(0)
    model = GPT(SMALL)
    model.eval()
    x = torch.randint(0, SMALL.vocab_size, (1, 16))
    y0, _ = model(x)

    qm = quantize_gpt(copy.deepcopy(model))
    qm.eval()
    y1, _ = qm(x)

    assert y1.shape == y0.shape
    assert (y1 - y0).norm() / y0.norm() < 0.05
    # tying preserved in int8: embedding and head share the SAME buffer (stored once)
    assert qm.wte.qweight.data_ptr() == qm.lm_head.qweight.data_ptr()
