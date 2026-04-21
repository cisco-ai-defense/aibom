# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# Multi-stage Dockerfile for cisco-aibom CLI.
# Includes ALL extras: analysis, security, agentic, cloud.
# Size ~800MB due to LLM and cloud SDK dependencies; the agentic
# extra is required because `analyze` invokes the LLM pipeline.

# ---------- Stage 1: Build ----------
FROM python:3.13-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN uv pip install --system --no-cache ".[all]"

# ---------- Stage 2: Runtime ----------
FROM python:3.13-slim AS runtime

LABEL org.opencontainers.image.title="cisco-aibom" \
      org.opencontainers.image.description="AI Bill of Materials CLI — detect AI assets in source code, container images, and repos with agentic enrichment and cloud scanning" \
      org.opencontainers.image.source="https://github.com/cisco-ai-defense/aibom" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="Cisco Systems, Inc."

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin/cisco-aibom /usr/local/bin/cisco-aibom

WORKDIR /workspace

ENTRYPOINT ["cisco-aibom"]
CMD ["--help"]
