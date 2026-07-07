"""Tokenize TinyStories into uint16 binary files for memmap training.

Usage (from the repo root):
    py -m train.dataset --train-mb 400
"""

import argparse
import os
import time

import numpy as np
from datasets import load_dataset

try:
    from tokenizer.bpe import BPETokenizer
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tokenizer.bpe import BPETokenizer

_FLUSH_TOKENS = 1_000_000


def write_split(texts, tok, path, limit_chars=None):
    """Encode stories (joined by blank lines) and stream uint16 ids to path."""
    n_tokens = 0
    n_chars = 0
    buf = []
    with open(path, "wb") as f:
        for text in texts:
            buf.extend(tok.encode(text + "\n\n"))
            n_chars += len(text) + 2
            if len(buf) >= _FLUSH_TOKENS:
                f.write(np.asarray(buf, dtype=np.uint16).tobytes())
                n_tokens += len(buf)
                buf = []
            if limit_chars is not None and n_chars >= limit_chars:
                break
        if buf:
            f.write(np.asarray(buf, dtype=np.uint16).tobytes())
            n_tokens += len(buf)
    return n_tokens, n_chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-mb", type=float, default=400.0,
                    help="chars of training text to tokenize (MB)")
    ap.add_argument("--val-mb", type=float, default=None,
                    help="cap validation text (MB); default: whole val split")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    assert len(tok.merges) > 0, "tokenizer has no merges — train it first"
    os.makedirs(args.out_dir, exist_ok=True)

    for split, out, limit in [
        ("train", f"{args.out_dir}/train.bin", int(args.train_mb * 1e6)),
        ("validation", f"{args.out_dir}/val.bin",
         int(args.val_mb * 1e6) if args.val_mb else None),
    ]:
        ds = load_dataset("roneneldan/TinyStories", split=split)
        t0 = time.time()
        n_tokens, n_chars = write_split((r["text"] for r in ds), tok, out, limit)
        print(f"{out}: {n_tokens:,} tokens from {n_chars:,} chars "
              f"({n_chars / max(n_tokens, 1):.2f} chars/token) in {time.time() - t0:,.0f}s")


if __name__ == "__main__":
    main()
