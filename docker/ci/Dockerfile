## CI image for optimum-rbln
##
## - Provides: Python 3.12 runtime + uv + common CLI tools used in workflows
## - Bakes-in: project dependencies from uv.lock (default + tests + quality groups)
## - Does NOT install the local project itself (so workflows can checkout and test any ref)
##
## Build example:
##   docker build -f docker/ci/Dockerfile -t optimum-rbln-ci:latest .
##
## Runtime example:
##   docker run --rm -it optimum-rbln-ci:latest python -c "import torch; print(torch.__version__)"

# syntax=docker/dockerfile:1.7

ARG UV_VERSION=0.9.18
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:3.12-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive

# Keep apt installs minimal but sufficient for common CI tasks.
# - git/curl/jq: used by workflows/scripts
# - build-essential/pkg-config: for any packages that may need compilation
# - libsndfile1/ffmpeg: used by audio-related Python deps (soundfile/librosa)
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    jq \
    openssh-client \
    build-essential \
    pkg-config \
    libsndfile1 \
    ffmpeg \
  && rm -rf /var/lib/apt/lists/*

# Copy uv (pinned) from the official image for reproducibility and faster builds.
COPY --from=uv /uv /bin/uv

# Configure a fixed venv path so workflows can rely on it.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Bake uv download cache into the image so CI jobs can reuse it without
# relying on GitHub Actions cache between jobs/runs.
ENV UV_CACHE_DIR=/opt/uv-cache

WORKDIR /workspace

# Copy only lockfiles first for better layer caching.
COPY pyproject.toml uv.lock /workspace/

# Install dependencies (default + tests + quality) according to uv.lock.
# We intentionally do NOT install the project itself, so the container can test arbitrary checkouts.
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_CACHE_DIR=/root/.cache/uv uv sync --frozen --group tests --group quality --no-install-project && \
    # Build backends for installing local checkout without network (PEP 517/660).
    UV_CACHE_DIR=/root/.cache/uv uv pip install hatchling hatch-vcs && \
    # Persist the populated uv cache into the image (runtime uses /opt/uv-cache).
    mkdir -p /opt/uv-cache && cp -a /root/.cache/uv/. /opt/uv-cache/ && \
    # Record the lockfile hash baked into this image (used by workflows to skip re-sync).
    sha256sum uv.lock > /opt/uv.lock.sha256


