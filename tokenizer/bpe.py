"""Byte-level BPE tokenizer for tiny-llm.

Training counts adjacent pairs over unique pre-tokenized chunks weighted by
frequency, with incremental pair-count updates after each merge — fast enough
to train a 4k vocab on ~50MB of text in pure Python.

Built with Claude Code.
"""

import json
import re
from collections import Counter, defaultdict

# A word keeps its single leading space (" cat"), so merges can learn
# space-prefixed word tokens; runs of extra whitespace form their own chunks.
# Chunks partition the text exactly, which is what makes decode lossless.
_CHUNK_RE = re.compile(r" ?\S+|\s+")

_NO_RANK = 1 << 60


def _pairs(ids):
    return zip(ids, ids[1:])


def _merge(ids, pair, new_id):
    """Replace every non-overlapping occurrence of pair, scanning left to right."""
    out = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self):
        self.merges = {}  # (id, id) -> new id; insertion order = merge rank
        self.vocab = {i: bytes([i]) for i in range(256)}
        self._encode_cache = {}

    # ------------------------------------------------------------- training

    def train(self, text, vocab_size):
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256")
        self.__init__()

        # unique chunk -> corpus frequency (finditer streams; no giant list)
        freqs = Counter(m.group() for m in _CHUNK_RE.finditer(text))
        words = [list(chunk.encode("utf-8")) for chunk in freqs]
        wfreq = list(freqs.values())

        pair_counts = Counter()
        pair_words = defaultdict(set)  # pair -> indices of words containing it
        for wi, ids in enumerate(words):
            f = wfreq[wi]
            for p in _pairs(ids):
                pair_counts[p] += f
                pair_words[p].add(wi)

        for _ in range(vocab_size - 256):
            if not pair_counts:
                break  # corpus fully merged; can't reach vocab_size
            # tie-break on the pair itself: deterministic regardless of
            # dict iteration order
            best = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            new_id = 256 + len(self.merges)
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

            for wi in list(pair_words[best]):
                old = words[wi]
                new = _merge(old, best, new_id)
                f = wfreq[wi]
                # delta-update from a full before/after recount of this word:
                # localized, so correctness only depends on _merge and _pairs
                oc = Counter(_pairs(old))
                nc = Counter(_pairs(new))
                for p, c in oc.items():
                    nkeep = nc.get(p, 0)
                    if nkeep != c:
                        pair_counts[p] += (nkeep - c) * f
                        if pair_counts[p] <= 0:
                            del pair_counts[p]
                    if nkeep == 0:
                        pair_words[p].discard(wi)
                for p, c in nc.items():
                    if p not in oc:
                        pair_counts[p] += c * f
                        pair_words[p].add(wi)
                words[wi] = new
            pair_words.pop(best, None)

    # ------------------------------------------------------------- encoding

    def encode(self, text):
        ids = []
        for m in _CHUNK_RE.finditer(text):
            chunk = m.group()
            cached = self._encode_cache.get(chunk)
            if cached is None:
                cached = self._encode_chunk(list(chunk.encode("utf-8")))
                self._encode_cache[chunk] = cached
            ids.extend(cached)
        return ids

    def _encode_chunk(self, ids):
        while len(ids) >= 2:
            # the adjacent pair whose merge was learned earliest, if any
            best = min(_pairs(ids), key=lambda p: self.merges.get(p, _NO_RANK))
            if best not in self.merges:
                break
            ids = _merge(ids, best, self.merges[best])
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    # ---------------------------------------------------------- persistence

    def save(self, path):
        data = {"merges": [[a, b, idx] for (a, b), idx in self.merges.items()]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        for a, b, idx in data["merges"]:  # saved in rank order
            tok.merges[(a, b)] = idx
            tok.vocab[idx] = tok.vocab[a] + tok.vocab[b]
        return tok
