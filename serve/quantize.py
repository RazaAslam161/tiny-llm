"""int8 per-output-channel weight quantization for Linear layers.

Per-channel (per output row) symmetric quantization: each row w_i gets its own
scale s_i = max(|w_i|) / 127, so a row is stored as int8 q = round(w / s) and
reconstructed as q * s. Per-channel beats per-tensor because one shared scale
would be dominated by the largest-magnitude row and crush the resolution of the
small ones. Weights dequantize to fp32 inside forward.

Built with Claude Code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_weight(w):
    """(out, in) fp32 -> (out, in) int8 quantized, (out,) fp32 per-row scale."""
    scale = (w.abs().amax(dim=1) / 127).clamp(min=1e-8)  # (out,)
    q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale


class QuantizedLinear(nn.Module):
    """Drop-in for nn.Linear holding int8 weights + fp32 per-channel scales."""

    def __init__(self, qweight, scale, bias=None):
        super().__init__()
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)
        self.bias = None if bias is None else nn.Parameter(bias)

    @classmethod
    def from_linear(cls, linear):
        q, scale = quantize_weight(linear.weight.data)
        bias = linear.bias.data.clone() if linear.bias is not None else None
        return cls(q, scale, bias)

    def forward(self, x):
        w = self.qweight.to(x.dtype) * self.scale[:, None]  # dequantize
        return F.linear(x, w, self.bias)


class QuantizedEmbedding(nn.Module):
    """int8 per-row (per-token) embedding table; lookup dequantizes on the fly."""

    def __init__(self, qweight, scale):
        super().__init__()
        self.register_buffer("qweight", qweight)  # (vocab, embd) int8
        self.register_buffer("scale", scale)      # (vocab,) fp32

    @classmethod
    def from_embedding(cls, emb):
        q, scale = quantize_weight(emb.weight.data)
        return cls(q, scale)

    def forward(self, idx):
        return self.qweight[idx].to(self.scale.dtype) * self.scale[idx].unsqueeze(-1)


def quantize_model_(module):
    """Recursively replace every nn.Linear in `module` with a QuantizedLinear."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, QuantizedLinear.from_linear(child))
        else:
            quantize_model_(child)
    return module


def quantize_gpt(model):
    """Full int8 quantization of a tiny-llm GPT, in place.

    Quantizes all block Linears and both embedding tables. The token embedding
    and the tied LM head share ONE quantized matrix (same int8 buffers), so the
    largest weight is stored once — that shared tying is what lets the checkpoint
    approach the ideal 4x fp32->int8 reduction instead of duplicating it.
    """
    for block in model.blocks:
        quantize_model_(block)

    # token embedding + tied output head: quantize once, share the buffers
    qw, scale = quantize_weight(model.wte.weight.data)
    model.wte = QuantizedEmbedding(qw, scale)
    model.lm_head = QuantizedLinear(qw, scale, bias=None)  # SAME tensors -> saved once

    model.wpe = QuantizedEmbedding.from_embedding(model.wpe)
    return model
