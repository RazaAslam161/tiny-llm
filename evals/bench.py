"""Benchmark naive vs KV-cached generation: tokens/sec and time-to-first-token
across context lengths, on CPU.

Naive generation recomputes attention over the whole prefix every step, so its
per-step cost grows with context (O(T^2) total). The cache makes each step
process a single token (O(T) total). This script measures the gap and writes a
plot + markdown table.

Usage (from the repo root):
    py -m evals.bench --gen 64 --out evals/kv_bench.png
"""

import argparse
import statistics
import time

import torch

from model.config import GPTConfig
from model.transformer import GPT
from serve.engine import generate_cached, generate_naive


def _time(fn, repeats):
    """Median wall time over `repeats` runs — single-shot CPU timings are noisy."""
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def bench_one(model, ctx_len, gen, vocab, repeats):
    prompt = torch.randint(0, vocab, (ctx_len,)).tolist()

    # time-to-first-token: cost of the first produced token (prompt processing)
    naive_ttft = _time(lambda: generate_naive(model, prompt, 1, temperature=0), repeats)
    cached_ttft = _time(lambda: generate_cached(model, prompt, 1, temperature=0), repeats)

    naive_total = _time(lambda: generate_naive(model, prompt, gen, temperature=0), repeats)
    cached_total = _time(lambda: generate_cached(model, prompt, gen, temperature=0), repeats)

    return {
        "ctx": ctx_len,
        "naive_tps": gen / naive_total,
        "cached_tps": gen / cached_total,
        "speedup": naive_total / cached_total,
        "naive_ttft_ms": naive_ttft * 1e3,
        "cached_ttft_ms": cached_ttft * 1e3,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=64, help="tokens to generate per run")
    ap.add_argument("--contexts", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--repeats", type=int, default=3, help="timed runs per point (median)")
    ap.add_argument("--out", default="evals/kv_bench.png")
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(torch.get_num_threads())
    # A dedicated benchmark model: real architecture width but a longer context
    # so 512-token prompts fit. Random weights — this measures speed, not quality.
    vocab = 4096
    cfg = GPTConfig(vocab_size=vocab, block_size=1024, n_layer=6, n_head=6, n_embd=384)
    model = GPT(cfg)
    model.eval()

    generate_cached(model, [1, 2, 3], 2, temperature=0)  # warm up lazy init

    print(f"benchmark model: block_size={cfg.block_size} (shipped model is 256); "
          f"gen={args.gen} tokens, median of {args.repeats} runs")
    rows = []
    for ctx in args.contexts:
        r = bench_one(model, ctx, args.gen, vocab, args.repeats)
        rows.append(r)
        print(f"ctx {r['ctx']:4d} | naive {r['naive_tps']:6.1f} tok/s | "
              f"cached {r['cached_tps']:7.1f} tok/s | {r['speedup']:5.1f}x | "
              f"TTFT naive {r['naive_ttft_ms']:6.1f}ms cached {r['cached_ttft_ms']:6.1f}ms")

    print("\n| context | naive tok/s | cached tok/s | speedup | naive TTFT | cached TTFT |")
    print("|--------:|------------:|-------------:|--------:|-----------:|------------:|")
    for r in rows:
        print(f"| {r['ctx']} | {r['naive_tps']:.1f} | {r['cached_tps']:.1f} | "
              f"{r['speedup']:.1f}x | {r['naive_ttft_ms']:.0f}ms | {r['cached_ttft_ms']:.0f}ms |")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\n(matplotlib not available — skipped {args.out})")
        return

    ctxs = [r["ctx"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(ctxs, [r["naive_tps"] for r in rows], "o-", label="naive")
    ax1.plot(ctxs, [r["cached_tps"] for r in rows], "o-", label="KV cached")
    ax1.set_xlabel("context length (tokens)")
    ax1.set_ylabel("generation throughput (tok/s)")
    ax1.set_title("Throughput vs context")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(ctxs, [r["speedup"] for r in rows], "o-", color="green")
    ax2.set_xlabel("context length (tokens)")
    ax2.set_ylabel("cached / naive speedup")
    ax2.set_title("KV-cache speedup")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
