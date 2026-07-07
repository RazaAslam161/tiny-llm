"""Validation perplexity and checkpoint size: fp32 vs int8-quantized.

Perplexity = exp(mean token cross-entropy). We evaluate both models on the SAME
val batches (same seed) so the delta is purely the quantization error.

Usage (from the repo root):
    py -m evals.perplexity --ckpt checkpoints/ckpt_final.pt --batches 40
"""

import argparse
import copy
import math
import os
import tempfile

import numpy as np
import torch

from model.config import GPTConfig
from model.transformer import GPT
from serve.quantize import quantize_gpt, quantize_model_


@torch.no_grad()
def perplexity(model, data, ctx, batches, batch_size, seed=0):
    model.eval()
    rng = np.random.default_rng(seed)
    total, n = 0.0, 0
    for _ in range(batches):
        ix = rng.integers(0, len(data) - ctx - 1, size=batch_size)
        x = torch.stack([torch.from_numpy(data[i:i + ctx].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + ctx].astype(np.int64)) for i in ix])
        _, loss = model(x, y)
        total += loss.item()
        n += 1
    avg = total / n
    return math.exp(avg), avg


def state_dict_bytes(model):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        path = f.name
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path)
    os.remove(path)
    return size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ckpt_final.pt")
    ap.add_argument("--data", default="data/val.bin")
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--scope", choices=["full", "linear"], default="full",
                    help="full = Linears + embeddings (tied, ~4x); linear = Linears only")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = GPTConfig(**ckpt["config"])
    fp32 = GPT(cfg)
    fp32.load_state_dict(ckpt["model"])

    quantize = quantize_gpt if args.scope == "full" else quantize_model_
    int8 = quantize(copy.deepcopy(fp32))

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    ctx = cfg.block_size

    ppl32, ce32 = perplexity(fp32, data, ctx, args.batches, args.batch_size)
    ppl8, ce8 = perplexity(int8, data, ctx, args.batches, args.batch_size)

    b32 = state_dict_bytes(fp32)
    b8 = state_dict_bytes(int8)

    print(f"fp32:  perplexity {ppl32:.3f}  (CE {ce32:.4f})  state_dict {b32/1e6:.1f} MB")
    print(f"int8:  perplexity {ppl8:.3f}  (CE {ce8:.4f})  state_dict {b8/1e6:.1f} MB")
    print(f"perplexity delta: {(ppl8 - ppl32) / ppl32 * 100:+.2f}%")
    print(f"size reduction:   {b32 / b8:.2f}x  ({(1 - b8 / b32) * 100:.1f}% smaller)")


if __name__ == "__main__":
    main()
