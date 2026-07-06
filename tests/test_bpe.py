"""Tests for tokenizer/bpe.py — byte-level BPE.

These tests pin down the interface they expect:

    class BPETokenizer:
        merges: dict[tuple[int, int], int]   # (id, id) -> new id; insertion order = merge order
        def train(self, text: str, vocab_size: int) -> None   # performs vocab_size - 256 merges
        def encode(self, text: str) -> list[int]
        def decode(self, ids: list[int]) -> str
"""
import random

import pytest

from tokenizer.bpe import BPETokenizer

# ASCII-only on purpose: the roundtrip cases below include Urdu and emoji,
# i.e. bytes the tokenizer never saw in training. Byte-level BPE must still
# encode them losslessly — that's the whole point of bytes over characters.
TRAIN_TEXT = (
    "One day, a little girl named Lily found a needle in her room. "
    "She knew it was difficult to play with it because it was sharp. "
    "The cat sat on the mat and the dog sat on the log. "
) * 5


@pytest.fixture(scope="module")
def trained():
    tok = BPETokenizer()
    tok.train(TRAIN_TEXT, vocab_size=300)
    return tok


ROUNDTRIP_CASES = [
    "",
    "a",
    "hello world",
    "One day, a little girl named Lily found a needle.",
    "   spaces\tand\nnewlines\r\n",
    "یہ ایک چھوٹی سی کہانی ہے",  # Urdu — Arabic script, 2-byte UTF-8 chars
    "🤖",                          # 4-byte emoji
    "👩🏽‍🚀",                       # emoji + skin-tone modifier + ZWJ
    "👨‍👩‍👧‍👦",                    # four emoji joined by zero-width joiners
    "🇵🇰",                          # flag: two regional-indicator codepoints
    "cat کہانی 🤖 story",           # mixed scripts in one string
]


@pytest.mark.parametrize("text", ROUNDTRIP_CASES)
def test_roundtrip(trained, text):
    assert trained.decode(trained.encode(text)) == text


def test_roundtrip_random_unicode(trained):
    rng = random.Random(20260707)
    chars = []
    while len(chars) < 3000:
        cp = rng.randrange(0x110000)
        if 0xD800 <= cp <= 0xDFFF:  # surrogates are not encodable as UTF-8
            continue
        chars.append(chr(cp))
    text = "".join(chars)
    assert trained.decode(trained.encode(text)) == text


def test_training_is_deterministic():
    a, b = BPETokenizer(), BPETokenizer()
    a.train(TRAIN_TEXT, vocab_size=280)
    b.train(TRAIN_TEXT, vocab_size=280)
    # Same corpus, same vocab size -> byte-identical merge rules, in the same order.
    assert list(a.merges.items()) == list(b.merges.items())
    sample = "One day the dog sat on the needle."
    assert a.encode(sample) == b.encode(sample)


def test_first_merges_hand_checked():
    # Corpus: "cat cat cats at" (bytes: c=99 a=97 t=116 s=115 space=32).
    #
    # Round 1 pair counts: (a,t)x4  (c,a)x3  (t,' ')x2  (' ',c)x2  rest x1
    #   -> merge 1: (97, 116) -> 256          vocab[256] = "at"
    # Round 2 recount:     (c,at)x3  (at,' ')x2  (' ',c)x2  rest x1
    #   -> merge 2: (99, 256) -> 257          vocab[257] = "cat"
    #
    # Holds with or without whitespace pre-tokenization: space-crossing
    # pairs never win a round here.
    tok = BPETokenizer()
    tok.train("cat cat cats at", vocab_size=258)
    assert list(tok.merges.items()) == [((97, 116), 256), ((99, 256), 257)]

    assert tok.encode("at") == [256]
    # encode must apply merges in training order: (a,t) first, then (c,at).
    assert tok.encode("cat") == [257]
    # bytes with no applicable merges pass through untouched
    assert tok.encode("dog") == [100, 111, 103]
    assert tok.decode([257, 115]) == "cats"


def test_vocab_size_256_means_zero_merges():
    tok = BPETokenizer()
    tok.train("abc abc", vocab_size=256)
    assert len(tok.merges) == 0
    assert tok.encode("abc") == [97, 98, 99]


def test_decode_invalid_utf8_replaces_instead_of_crashing():
    # A language model can emit any id sequence, including ones whose bytes
    # are not valid UTF-8. id 195 is byte 0xC3 — the first half of a 2-byte
    # character with no continuation byte. decode should map it to U+FFFD
    # (errors="replace", same convention as GPT-2), not raise.
    tok = BPETokenizer()
    tok.train("cat cat cats at", vocab_size=258)
    assert tok.decode([195]) == "�"
