# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# Multi-stage Dockerfile for cisco-aibom CLI (deterministic scan).
# Includes [analysis] and [security] extras; excludes agentic/cloud SDKs.
# See Dockerfile.agentic for the full variant.

# ---------- Stage 1: Build ----------
FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN uv pip install --system --no-cache ".[analysis,security]"

# ---------- Stage 2: Runtime ----------
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="cisco-aibom" \
      org.opencontainers.image.description="AI Bill of Materials CLI — detect AI assets in source code" \
      org.opencontainers.image.source="https://github.com/cisco-ai-defense/aibom" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="Cisco Systems, Inc."

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/cisco-aibom /usr/local/bin/cisco-aibom

WORKDIR /workspace

ENTRYPOINT ["cisco-aibom"]
CMD ["--help"]
