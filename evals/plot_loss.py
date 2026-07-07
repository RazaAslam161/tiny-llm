"""Plot the training/validation loss curve from a training CSV log.

Usage (from the repo root):
    py -m evals.plot_loss --csv checkpoints/loss_log.csv --out evals/loss_curve.png
"""

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="checkpoints/loss_log.csv")
    ap.add_argument("--out", default="evals/loss_curve.png")
    args = ap.parse_args()

    steps, train, val = [], [], []
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            train.append(float(row["train_loss"]))
            val.append(float(row["val_loss"]))

    plt.figure(figsize=(8, 5))
    plt.plot(steps, train, label="train", alpha=0.6, linewidth=1)
    plt.plot(steps, val, label="val", linewidth=2)
    plt.axhline(1.5, color="gray", linestyle="--", linewidth=0.8, label="target 1.5")
    plt.xlabel("step")
    plt.ylabel("cross-entropy loss")
    plt.title(f"tiny-llm training — final val {val[-1]:.3f}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"saved {args.out}  (final train {train[-1]:.3f}, val {val[-1]:.3f})")


if __name__ == "__main__":
    main()
