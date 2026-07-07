"""Export a small, weights-only checkpoint for deployment.

The training checkpoint carries optimizer state (~100 MB of it). Serving only
needs the model weights + config, so this strips the rest.

Usage (from the repo root):
    py -m serve.export --ckpt checkpoints/ckpt_final.pt --out checkpoints/model.pt
"""

import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ckpt_final.pt")
    ap.add_argument("--out", default="checkpoints/model.pt")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    torch.save({"model": ckpt["model"], "config": ckpt["config"]}, args.out)
    print(f"saved {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB) from {args.ckpt}")


if __name__ == "__main__":
    main()
