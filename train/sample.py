"""Generate text from a trained checkpoint.

Usage (from the repo root):
    py -m train.sample --ckpt checkpoints/ckpt_final.pt --prompt "Once upon a time"
"""

import argparse

import torch

try:
    from model.config import GPTConfig
    from model.transformer import GPT
    from tokenizer.bpe import BPETokenizer
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model.config import GPTConfig
    from model.transformer import GPT
    from tokenizer.bpe import BPETokenizer


@torch.no_grad()
def generate(model, ids, max_new_tokens, temperature=0.8, top_k=50):
    model.eval()
    idx = torch.tensor([ids], dtype=torch.long)
    for _ in range(max_new_tokens):
        window = idx[:, -model.config.block_size:]
        logits, _ = model(window)
        logits = logits[:, -1, :]
        if temperature <= 0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                kth = torch.topk(logits, top_k).values[:, -1, None]
                logits[logits < kth] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, nxt], dim=1)
    return idx[0].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ckpt_final.pt")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    tok = BPETokenizer.load(args.tokenizer)

    out = generate(model, tok.encode(args.prompt), args.max_new_tokens,
                   args.temperature, args.top_k)
    print(tok.decode(out))


if __name__ == "__main__":
    main()
