"""Tests for the FastAPI server, using a tiny random model + the real tokenizer
so they run without the trained checkpoint."""

import json
from pathlib import Path

import pytest
import torch

from model.config import GPTConfig
from model.transformer import GPT
from serve.server import ByteStreamDecoder, build_app
from tokenizer.bpe import BPETokenizer

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TOKENIZER = Path(__file__).resolve().parents[1] / "tokenizer" / "tokenizer.json"


@pytest.fixture(scope="module")
def client():
    torch.manual_seed(0)
    tok = BPETokenizer.load(str(TOKENIZER))
    model = GPT(GPTConfig(vocab_size=4096, block_size=64, n_layer=2, n_head=2, n_embd=64))
    model.eval()
    return fastapi_testclient.TestClient(build_app(model, tok))


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "tiny-llm" in r.text


def test_generate_streams_sse_tokens(client):
    r = client.post("/generate", json={"prompt": "Once upon a time", "max_tokens": 20})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    events = [ln for ln in r.text.split("\n\n") if ln.startswith("data: ")]
    assert events[-1] == "data: [DONE]"
    tokens = [json.loads(e[6:])["token"] for e in events[:-1] if "token" in e]
    assert tokens  # at least one token streamed
    assert all(isinstance(t, str) for t in tokens)


def test_bytestream_decoder_holds_split_multibyte():
    dec = ByteStreamDecoder()
    smiley = "🙂".encode("utf-8")  # 4 bytes; feed them split across "tokens"
    assert dec.push(smiley[:2]) == ""     # incomplete — nothing emitted yet
    assert dec.push(smiley[2:]) == "🙂"    # completes the character
    # ascii passes straight through
    assert dec.push(b"hi") == "hi"


def test_bytestream_decoder_flush_replaces_leftover():
    dec = ByteStreamDecoder()
    assert dec.push(b"\xff") == ""       # invalid/incomplete, held
    assert dec.flush() == "�"       # emitted as replacement char at the end
