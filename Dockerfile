# Deploy image for Hugging Face Spaces (SDK: docker). Serves the FastAPI app.
FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch + the minimal serving deps (no datasets/matplotlib/pytest)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        torch numpy fastapi "uvicorn[standard]"

COPY model/ model/
COPY tokenizer/ tokenizer/
COPY serve/ serve/
COPY web/ web/
COPY checkpoints/model.pt checkpoints/model.pt

# quantize in memory on startup to shrink the RAM footprint (+0.03% perplexity)
ENV TINY_LLM_CKPT=checkpoints/model.pt \
    TINY_LLM_QUANTIZE=1

EXPOSE 7860
CMD ["uvicorn", "serve.server:app", "--host", "0.0.0.0", "--port", "7860"]
