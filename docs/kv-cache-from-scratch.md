# The KV cache, and why it took me a while to get right

Generating text one token at a time is wasteful by default, and the fix is a
key/value cache. I built one for [tiny-llm](https://github.com/RazaAslam161/tiny-llm)
(a 12M GPT I trained on TinyStories) and got about a 7.4x speedup at the model's
context length. Here's the part that actually took some thinking.

## the wasteful part

To predict token t+1 the model runs the whole sequence so far through every
layer and reads off the last position:

```python
for _ in range(n):
    logits = model(all_tokens)      # runs the ENTIRE prefix
    nxt = sample(logits[:, -1])     # only uses the last row
    all_tokens.append(nxt)
```

Every step redoes the whole prefix even though the earlier tokens haven't
changed. Generating T tokens is O(T^2). You can watch it happen: on CPU my model
drops from 21 tok/s at 64 tokens of context to 4 tok/s at 512.

## what actually changes each step

Inside each attention layer, when you add a token:

- the new token computes its own query, key, value
- it attends over all the previous keys and values plus its own
- the previous keys and values are exactly what they were last step (the causal
  mask means a token never depends on later ones)

So the old K/V are pure recomputation. Cache them. Now each step only computes
K/V for the new token, appends to the cache, and attends against the whole thing:

```python
logits, kv = model(prompt, use_cache=True)   # prefill once
for _ in range(n):
    nxt = sample(logits[:, -1])
    logits, kv = model(nxt, past_kv=kv, use_cache=True)   # one token in
```

Per-step work goes from O(prefix) to O(1). Queries don't get cached, there's only
ever one query (the new token). Only K and V pile up.

## the part I kept getting wrong

A cache is only allowed to be faster, not different. So the test isn't "looks
fine", it's that cached generation gives the same logits as a full forward at
every position:

```python
full = model(seq)
for t, tok in enumerate(seq):
    logits, kv = model(tok, past_kv=kv, use_cache=True)
    assert torch.allclose(logits, full[:, t], atol=1e-5)
```

Getting that to pass came down to the causal mask. With a cache the T new query
rows attend over T_past + T keys, and the mask has to say "query i can see key j
if j is at or before i's actual position". That works out to the last T rows of
the full triangular mask, sliced out. Proving it for both the prefill (many
tokens at once) and the decode step (one token) is where the real understanding
sits.

## the thing tutorials skip

My model uses learned absolute position embeddings. Position 0 has a vector,
position 1 has another, up to the context length. That does not play nice with a
cache once you go past the window.

The naive path handles overflow with a sliding window: it re-embeds the last N
tokens starting from position 0 each step. A cache can't copy that. The stored
K/V already have their positions baked in, and you can't shift them without
recomputing, which is the whole thing you were trying to skip.

So a sliding-window KV cache doesn't work with absolute positions, full stop. The
two paths can only match inside the context window. This is why real models use
relative position schemes like RoPE and ALiBi: relative positions ride along with
the cached K/V, so the window slides cleanly. I found this out by hitting the
wall, which stuck better than reading it would have.

## the numbers

| ctx | naive | cached | speedup |
|--:|--:|--:|--:|
| 64 | 21.6 tok/s | 69.2 tok/s | 3.2x |
| 128 | 15.5 | 74.7 | 4.8x |
| 256 | 8.1 | 65.4 | 7.4x |
| 512 | 4.3 | 48.7 | 11.2x |

The speedup grows with context, which is the point, that's when generation
actually hurts. Time to first token is the same for both (they each process the
whole prompt once), the cache only helps the tokens after that.

If you can derive the mask slice, say why queries aren't cached, and explain the
absolute-position limitation, you've got the main trick in LLM serving.
