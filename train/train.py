"""Training loop for tiny-llm.

Usage (from the repo root):
    py -m train.train --steps 30000 --batch-size 64 --ctx 256

AdamW + linear warmup into cosine decay, gradient clipping, mixed precision
on CUDA, periodic checkpoints, and a CSV loss log.
"""

import argparse
import csv
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch

try:
    from model.config import GPTConfig
    from model.transformer import GPT
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model.config import GPTConfig
    from model.transformer import GPT


def get_batch(data, batch_size, ctx, device, rng):
    ix = rng.integers(0, len(data) - ctx - 1, size=batch_size)
    x = torch.stack([torch.from_numpy(data[i:i + ctx].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + ctx].astype(np.int64)) for i in ix])
    if device == "cuda":
        return x.pin_memory().to(device, non_blocking=True), \
               y.pin_memory().to(device, non_blocking=True)
    return x, y


def lr_at(step, base_lr, warmup, max_steps):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, max_steps - warmup)
    return 0.1 * base_lr + 0.45 * base_lr * (1 + math.cos(math.pi * t))


@torch.no_grad()
def eval_loss(model, data, batch_size, ctx, device, rng, iters=50):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, batch_size, ctx, device, rng)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--eval-interval", type=int, default=500)
    ap.add_argument("--ckpt-interval", type=int, default=1000)
    ap.add_argument("--resume", default=None, help="checkpoint path to resume from")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_data = np.memmap(f"{args.data_dir}/train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(f"{args.data_dir}/val.bin", dtype=np.uint16, mode="r")
    print(f"device={device}  train={len(train_data):,} tokens  val={len(val_data):,} tokens")

    config = GPTConfig(block_size=args.ctx)
    model = GPT(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        print(f"resumed from {args.resume} at step {start_step}")

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    autocast = torch.autocast(device, dtype=amp_dtype) if use_amp else nullcontext()
    scaler = torch.amp.GradScaler(device, enabled=use_amp and amp_dtype == torch.float16)

    log_path = os.path.join(args.out_dir, "loss_log.csv")
    if start_step == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "train_loss", "val_loss", "lr"])

    rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(args.seed + 1)
    model.train()
    t0 = time.time()

    def save(step, name):
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "step": step, "config": config.__dict__},
                   os.path.join(args.out_dir, name))

    for step in range(start_step, args.steps):
        lr = lr_at(step, args.lr, args.warmup, args.steps)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.ctx, device, rng)
        with autocast:
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if step % args.eval_interval == 0 or step == args.steps - 1:
            vl = eval_loss(model, val_data, args.batch_size, args.ctx, device, eval_rng)
            tps = args.batch_size * args.ctx * args.eval_interval / max(time.time() - t0, 1e-9)
            print(f"step {step:6d}  train {loss.item():.4f}  val {vl:.4f}  "
                  f"lr {lr:.2e}  {tps:,.0f} tok/s")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step, f"{loss.item():.4f}", f"{vl:.4f}", f"{lr:.2e}"])
            t0 = time.time()

        if step > 0 and step % args.ckpt_interval == 0:
            save(step, "ckpt.pt")

    save(args.steps - 1, "ckpt_final.pt")
    print(f"done — checkpoints in {args.out_dir}/")


if __name__ == "__main__":
    main()
