## CI image for optimum-rbln
##
## Python 3.12 + uv + the CLI tools the remaining GHA workflows shell out to.
## The jobs that run here install what they need themselves: deploy.yaml syncs its
## own group, auto_compiler_update only curls and jqs.
##
## Build example:
##   docker build -f docker/ci/Dockerfile -t optimum-rbln-ci:latest .

# syntax=docker/dockerfile:1.7

ARG UV_VERSION=0.11.33
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:3.12-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive

# What the workflows shell out to. The compilers and the audio libraries went with
# the dependencies they were for: nothing installed here builds an extension.
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    jq \
    openssh-client \
  && rm -rf /var/lib/apt/lists/*

# Copy uv (pinned) from the official image for reproducibility and faster builds.
COPY --from=uv /uv /bin/uv

WORKDIR /workspace
