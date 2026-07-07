"""Token sampling: temperature, top-k, and top-p (nucleus), by hand.

Given last-position logits (B, vocab) the pipeline is:
  temperature -> top-k filter -> top-p filter -> softmax -> multinomial

temperature <= 0 short-circuits to greedy argmax. Each filter sets rejected
logits to -inf so the final softmax renormalizes over the survivors only.

Built with Claude Code.
"""

import torch
import torch.nn.functional as F


def sample(logits, temperature=1.0, top_k=0, top_p=1.0, generator=None):
    """Return next-token ids of shape (B, 1) from logits of shape (B, vocab)."""
    if temperature <= 0:  # greedy
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k).values[:, -1, None]  # (B, 1)
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p < 1.0:
        # sort descending, walk the cumulative distribution, keep the smallest
        # prefix whose mass reaches p (always at least the top token)
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumprobs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumprobs > top_p
        remove[..., 1:] = remove[..., :-1].clone()  # shift: keep the crossing token
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)
