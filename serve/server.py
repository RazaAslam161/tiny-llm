"""FastAPI app, streams tokens over SSE.

    POST /generate  {prompt, max_tokens, temperature, top_p, top_k} -> event-stream
    GET  /healthz
    GET  /          web UI

build_app() takes the model + tokenizer so tests can pass a small one; the
module-level app loads the real checkpoint (paths and quantize flag from env).
"""

import json
import os
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from model.config import GPTConfig
from model.transformer import GPT
from serve.engine import stream_tokens
from serve.quantize import quantize_gpt
from tokenizer.bpe import BPETokenizer

_WEB = Path(__file__).resolve().parents[1] / "web" / "index.html"


class GenRequest(BaseModel):
    prompt: str = "Once upon a time"
    max_tokens: int = 200
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50


class ByteStreamDecoder:
    """Turn a stream of per-token byte strings into text, emitting only complete
    UTF-8 characters. A multi-byte char split across tokens is held until its
    bytes arrive, so the client never sees a transient replacement char."""

    def __init__(self):
        self.buf = b""

    def push(self, token_bytes):
        self.buf += token_bytes
        try:
            text = self.buf.decode("utf-8")
            self.buf = b""
            return text
        except UnicodeDecodeError as e:
            text = self.buf[:e.start].decode("utf-8")  # complete chars only
            self.buf = self.buf[e.start:]              # keep the incomplete tail
            return text

    def flush(self):
        text = self.buf.decode("utf-8", errors="replace")
        self.buf = b""
        return text


def build_app(model, tok):
    app = FastAPI(title="tiny-llm")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "ready": model is not None}

    @app.get("/", response_class=HTMLResponse)
    def index():
        if _WEB.exists():
            return _WEB.read_text(encoding="utf-8")
        return "<h1>tiny-llm</h1><p>web/index.html not found</p>"

    @app.post("/generate")
    def generate(req: GenRequest):
        if model is None:
            return StreamingResponse(
                iter([f"data: {json.dumps({'error': 'model not loaded'})}\n\n"]),
                media_type="text/event-stream", status_code=503,
            )

        def event_stream():
            ids = tok.encode(req.prompt)
            decoder = ByteStreamDecoder()
            for tid in stream_tokens(model, ids, req.max_tokens,
                                     req.temperature, req.top_k, req.top_p):
                piece = decoder.push(tok.vocab[tid])
                if piece:
                    yield f"data: {json.dumps({'token': piece})}\n\n"
            tail = decoder.flush()
            if tail:
                yield f"data: {json.dumps({'token': tail})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


def _load_default():
    ckpt_path = os.environ.get("TINY_LLM_CKPT", "checkpoints/ckpt_final.pt")
    tok_path = os.environ.get("TINY_LLM_TOKENIZER", "tokenizer/tokenizer.json")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    if os.environ.get("TINY_LLM_QUANTIZE") == "1":
        quantize_gpt(model)
    return model, BPETokenizer.load(tok_path)


try:
    _model, _tok = _load_default()
except Exception as exc:  # missing checkpoint etc. — app still serves /healthz
    print(f"[server] model not loaded: {exc}")
    _model, _tok = None, None

app = build_app(_model, _tok)
