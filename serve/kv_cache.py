"""Per-layer K/V cache. The model returns the keys/values it attended over, we
stash them and feed them back so the next step only computes K/V for the new
token instead of redoing the whole prefix.
"""


class KVCache:
    # layers: one (K, V) pair per block, each (B, n_head, tokens, head_dim).
    # None until the first (prefill) forward fills it.

    def __init__(self):
        self.layers = None

    @property
    def length(self):
        return 0 if self.layers is None else self.layers[0][0].size(2)

    def update(self, presents):
        self.layers = presents
        return self
