"""Train the byte-level BPE tokenizer on a TinyStories slice.

Usage (from the repo root):
    py -m tokenizer.train_tokenizer --mb 50 --vocab-size 4096
"""

import argparse
import time

from datasets import load_dataset

try:
    from tokenizer.bpe import BPETokenizer
except ImportError:  # run as a plain script from inside tokenizer/
    from bpe import BPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=float, default=50.0, help="corpus slice size in MB")
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--out", default="tokenizer/tokenizer.json")
    args = ap.parse_args()

    print(f"Collecting ~{args.mb:.0f}MB of TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    target = int(args.mb * 1e6)
    parts, total = [], 0
    for row in ds:
        t = row["text"]
        parts.append(t)
        total += len(t) + 1
        if total >= target:
            break
    text = "\n".join(parts)
    print(f"Corpus: {len(text):,} chars, {len(parts):,} stories")

    tok = BPETokenizer()
    t0 = time.time()
    tok.train(text, vocab_size=args.vocab_size)
    elapsed = time.time() - t0
    print(f"Trained {len(tok.merges):,} merges in {elapsed:,.1f}s")

    tok.save(args.out)
    print(f"Saved {args.out}")

    # Gate metric on held-out data: roadmap wants ~1.2-1.5 tokens per word.
    val = load_dataset("roneneldan/TinyStories", split="validation")
    sample = "\n".join(val[i]["text"] for i in range(200))
    ids = tok.encode(sample)
    assert tok.decode(ids) == sample, "roundtrip failed on validation sample"
    ratio = len(ids) / len(sample.split())
    print(f"Validation sample: {len(ids):,} tokens / {len(sample.split()):,} words "
          f"= {ratio:.3f} tokens per word")

    print("\nSample encodings:")
    for s in ["Once upon a time", "The little girl smiled."]:
        e = tok.encode(s)
        print(f"  {s!r} -> {len(e)} tokens: {[tok.vocab[i].decode('utf-8', 'replace') for i in e]}")


if __name__ == "__main__":
    main()
