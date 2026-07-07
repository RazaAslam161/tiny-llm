# Deploying tiny-llm to Hugging Face Spaces

The Space serves the FastAPI app in a Docker container on the free CPU tier. The quantized
12M model is small and responsive there.

## One-time setup

1. **Export the deploy checkpoint** (weights only, ~49 MB — no optimizer state):
   ```bash
   python -m serve.export --ckpt checkpoints/ckpt_final.pt --out checkpoints/model.pt
   ```

2. **Create the Space**: on huggingface.co → New → Space → **SDK: Docker** → Blank. Name it
   `tiny-llm`. This creates a git repo at `https://huggingface.co/spaces/<username>/tiny-llm`.

3. **Add the Space README header.** The Space's `README.md` must start with this YAML block
   (copy `deploy/space-README.md` from this repo to the Space root as `README.md`):
   ```yaml
   ---
   title: tiny-llm
   emoji: 📖
   colorFrom: blue
   colorTo: indigo
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```

4. **Push the files** to the Space repo:
   ```bash
   git clone https://huggingface.co/spaces/<username>/tiny-llm hf-space
   cd hf-space
   # copy from this repo:
   cp -r ../model ../tokenizer ../serve ../web ../Dockerfile ../.dockerignore .
   mkdir -p checkpoints && cp ../checkpoints/model.pt checkpoints/
   cp ../deploy/space-README.md README.md
   git lfs install && git lfs track "*.pt"
   git add -A && git commit -m "deploy tiny-llm" && git push
   ```

   HF builds the Docker image and starts the container. First build takes a few minutes
   (installing torch). When it's live, the URL streams generations just like the local demo.

5. **Add the live link** to the main repo README (there's a commented-out placeholder near
   the top).

## Notes

- The container runs `uvicorn serve.server:app --port 7860` with `TINY_LLM_QUANTIZE=1`, so
  it quantizes in memory on startup for a smaller footprint.
- `*.pt` must be tracked by git-lfs on the Space (step 4) — HF rejects large plain-git blobs.
- Free CPU Spaces sleep when idle and cold-start on the next request; that's expected.
