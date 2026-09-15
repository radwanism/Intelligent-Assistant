# Telecom Egypt Intelligent Assistant — CPU / on-premises image.
#
# One image runs all three services; the command selects which. They share the
# same code and model cache, so building three images would triple the layer
# storage for no isolation benefit — the isolation that matters is the process
# boundary at runtime (see docker-compose.yml), not the image.
#
# NOTE: this Dockerfile has NOT been built or run. Docker is not installed on the
# development machine and Colab cannot run nested containers, so it is provided
# as a deployment artifact rather than a verified one. Treat the first `docker
# compose build` as the real test.

FROM python:3.11-slim-bookworm

# ffmpeg is a hard requirement: faster-whisper decodes audio through it.
# libgomp1 is needed by onnxruntime and ctranslate2 for OpenMP threading.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv gives the same resolver and lockfile as local development, so the container
# and the laptop install byte-identical dependency versions.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    # Keep model weights on a mounted volume, never baked into the image:
    # the GGUF alone is ~2 GB and would make the image unshippable.
    HF_HOME=/models/hf \
    TE_DATA_DIR=/data

WORKDIR /app

# Dependencies first, in their own layer, so editing source does not reinstall
# 296 packages on every rebuild.
COPY pyproject.toml uv.lock README.md ./
# The image targets the on-premises CPU profile, so it installs the `cpu` extra
# (llama.cpp). Its prebuilt wheels come from the project's own index; on Linux
# a source build would also work but costs several minutes of layer time.
RUN uv pip install --system --no-cache \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
        --index-strategy unsafe-best-match \
        -e ".[cpu]"

COPY src/ ./src/
COPY scripts/ ./scripts/
RUN uv pip install --system --no-cache --no-deps -e .

# Non-root: the services never need to write outside /data.
RUN useradd --create-home --uid 1000 assistant \
    && mkdir -p /data /models \
    && chown -R assistant:assistant /app /data /models
USER assistant

EXPOSE 8000 8001 7860

# Defaults to the core service; compose overrides for speech and ui.
CMD ["python", "-m", "te_assistant.api"]
