# What I learned implementing a KV cache from scratch

Autoregressive text generation has an embarrassing inefficiency at its core, and the fix —
a key/value cache — is one of the most-asked topics in LLM-systems interviews. I built one
from scratch for [tiny-llm](https://github.com/RazaAslam161/tiny-llm), a 12M-parameter GPT
I trained on TinyStories, and measured a **7.4× speedup** at the model's context length.
Here's what the exercise actually teaches you.

## The O(T²) problem

To generate token *t+1*, a transformer runs the whole sequence so far through every layer
and reads the prediction off the last position. The naive loop looks like this:

```python
for _ in range(max_new_tokens):
    logits = model(all_tokens_so_far)     # processes the ENTIRE prefix
    next_token = sample(logits[:, -1, :]) # ...but only uses the last row
    all_tokens_so_far.append(next_token)
```

Every step reprocesses the entire prefix, even though positions `0..t-1` are unchanged.
Generating *T* tokens costs O(T²) total work. You can watch it happen: on CPU my model's
throughput collapses from 21.6 tok/s at 64 tokens of context to 4.3 tok/s at 512.

## What actually changes each step — and what doesn't

Here's the insight the cache is built on. When you add one token, inside each attention
layer:

- The new token computes its own **query, key, and value**.
- It attends over **all previous keys and values** plus its own.
- Every *previous* token's key and value are **exactly what they were last step** — they
  don't depend on the new token (that's what the causal mask guarantees).

So the keys and values are pure recomputation. Cache them. Each step then only computes K/V
for the single new token, appends to the cache, and attends against the whole cache:

```python
logits, kv = model(prompt, use_cache=True)   # prefill once
for _ in range(max_new_tokens):
    next_token = sample(logits[:, -1, :])
    logits, kv = model(next_token, past_kv=kv, use_cache=True)  # ONE token in
```

Per-step work drops from O(prefix) to O(1); total generation goes O(T²) → O(T). Note that
**queries are not cached** — each step has exactly one query (the new token). Only keys and
values accumulate.

## The correctness bar: identical, not just similar

A cache is a pure optimization. If it changes the output, it's a bug, not a speedup. So the
real test isn't "does it look right" — it's that cached generation produces **bit-identical
logits** to the naive path at every position:

```python
full_logits = model(sequence)                 # one full forward
for t, token in enumerate(sequence):
    logits, kv = model(token, past_kv=kv, use_cache=True)
    assert torch.allclose(logits, full_logits[:, t, :], atol=1e-5)
```

Getting this to pass forces you to get the details exactly right — chiefly the causal mask.
With a cache, the *T* new query rows attend over `T_past + T` keys, and the mask has to say
"query *i* may attend to key *j* iff *j* ≤ its own absolute position." It turns out this is
just the **last T rows** of the full causal mask — reusing the same triangular buffer,
sliced. Deriving that slice, and proving it holds for both the multi-token prefill and the
one-token decode step, is where the understanding lives.

## The limitation nobody mentions

Here's the thing the tutorials skip. My model uses **learned absolute position embeddings**
— position 0 gets one vector, position 1 another, up to the context length. That composes
badly with a cache once you exceed the context window.

The naive path handles overflow with a sliding window: it re-embeds the last *N* tokens
starting from position 0 each step. But a cache **can't** replicate that — the stored keys
and values have their original positions already baked in, and you can't retroactively shift
them without recomputing, which defeats the entire point.

So a sliding-window KV cache is fundamentally incompatible with absolute position
embeddings. The two paths can only be guaranteed identical *within* the context window. This
isn't a wart to hide — it's exactly **why production models use relative position schemes
like RoPE and ALiBi**: relative positions travel with the cached K/V, so the cache slides
cleanly. Discovering this by hitting the wall yourself is worth more than reading it.

## The payoff

| context | naive | cached | speedup |
|--:|--:|--:|--:|
| 64 | 21.6 tok/s | 69.2 tok/s | 3.2× |
| 128 | 15.5 | 74.7 | 4.8× |
| 256 | 8.1 | 65.4 | **7.4×** |
| 512 | 4.3 | 48.7 | 11.2× |

The speedup **grows with context** — which is the whole point, because that's exactly when
generation hurts. And time-to-first-token is unchanged: the cache does nothing for the first
token (both paths process the full prompt once); it only pays off on every token after.

That's the shape of the O(T²) → O(T) win, made concrete. If you can derive the mask slice,
explain why queries aren't cached, and articulate the absolute-position limitation, you
understand the single most important trick in LLM serving.

---

*Part of [tiny-llm](https://github.com/RazaAslam161/tiny-llm) — a 12M-parameter GPT trained
from scratch with a hand-built inference stack.*
