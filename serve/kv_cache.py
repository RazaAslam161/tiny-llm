"""Key/value cache for autoregressive generation.

During generation the model returns, per layer, the full (K, V) it attended
over. Instead of recomputing those from the whole prefix every step, we stash
them here and feed them back so each new token only computes its own K/V and
attends against the cache. That turns per-step work from O(prefix) to O(1),
and total generation from O(T^2) to O(T).

Built with Claude Code.
"""


class KVCache:
    """Holds the per-layer (K, V) tensors produced so far.

    `layers` is a list with one (K, V) pair per transformer block, each shaped
    (B, n_head, tokens_so_far, head_dim). It is None until the first forward
    pass (the prefill) populates it.
    """

    def __init__(self):
        self.layers = None

    @property
    def length(self):
        """Number of tokens currently cached (0 before prefill)."""
        return 0 if self.layers is None else self.layers[0][0].size(2)

    def update(self, presents):
        self.layers = presents
        return self
