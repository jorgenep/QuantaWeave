# QuantaWeave — CUDA image (the default backend). Build and run:
#   docker build -t quantweave .
#   docker run --gpus all -v $(pwd)/artifacts:/app/artifacts -v $(pwd)/data:/app/data quantweave \
#       python src/train_quantweave_moe.py --data data/smoke/tinystories.jsonl --steps 1000
#   docker run --gpus all -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts quantweave \
#       python src/serve_quantweave.py --host 0.0.0.0 --latest --api-key "$MOE_API_KEY"
#
# For ROCm or Intel XPU, swap the base image and the torch install line for the matching build — see README's
# "Device Backends" section — everything else here is backend-agnostic. For a CPU-only image, drop the CUDA base
# for plain `python:3.12-slim` and the torch install line for a CPU wheel.
#
# NOTE: this Dockerfile has not been built or run in this environment (no Docker available here) — it follows the
# same install steps documented and exercised in README/PARAMETERS.md, but treat it as unverified until you have
# built it yourself, the same honesty this project applies to anything untestable on the machine it was written on.

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# dependencies first, so code-only changes don't invalidate the pip layer
COPY pyproject.toml ./
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch

COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
RUN pip install --no-cache-dir -e ".[data,tools,hf,export,serve]"

# artifacts/ (checkpoints, exports, run archives) and data/ are meant to be mounted volumes, not baked into the
# image: they hold your run state and training data, not anything this image should own or ship with.
VOLUME ["/app/artifacts", "/app/data"]

EXPOSE 8000

ENTRYPOINT ["python3"]
CMD ["src/serve_quantweave.py", "--host", "0.0.0.0", "--latest"]
