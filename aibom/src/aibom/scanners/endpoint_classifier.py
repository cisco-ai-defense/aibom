# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Classify endpoint URLs into ``llm_endpoint``, ``model_endpoint``, or ``vector_store``.

Uses five signals in priority order:

1. Vector-store URL tokens (always wins).
2. Cloud LLM provider URL pattern (unambiguous platform signal).
3. Paired model identity via 3-tier model registry (LiteLLM + builtin + HF).
4. Config-key context heuristic (plain string containment).
5. Fallback: ``model_endpoint`` (self-hosted serving, unknown providers).
"""

from __future__ import annotations

import re

from .model_detector import registry_lookup

# Cloud LLM provider URL patterns (regex only for domain matching).
# AWS Bedrock covers all partitions: commercial (.amazonaws.com),
# GovCloud (us-gov-* regions, same suffix), FIPS (bedrock-runtime-fips),
# and China (.amazonaws.com.cn).
_CLOUD_LLM_PROVIDER_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("azure_openai", re.compile(r"\.openai\.azure\.com", re.I)),
    ("openai", re.compile(r"api\.openai\.com", re.I)),
    ("anthropic", re.compile(r"api\.anthropic\.com", re.I)),
    ("aws_bedrock", re.compile(
        r"bedrock-runtime(?:-fips)?\.[a-z0-9-]+\.amazonaws\.com(?:\.cn)?", re.I,
    )),
    ("google_vertex", re.compile(r"[a-z0-9-]+-aiplatform\.googleapis\.com", re.I)),
    ("cohere", re.compile(r"api\.cohere\.(com|ai)", re.I)),
    ("mistral", re.compile(r"api\.mistral\.ai", re.I)),
    ("groq", re.compile(r"api\.groq\.com", re.I)),
    ("together", re.compile(r"api\.together\.xyz", re.I)),
    ("fireworks", re.compile(r"api\.fireworks\.ai", re.I)),
    ("deepinfra", re.compile(r"api\.deepinfra\.com", re.I)),
    ("perplexity", re.compile(r"api\.perplexity\.ai", re.I)),
]

_VECTOR_STORE_URL_TOKENS = ("weaviate", "pinecone", "qdrant", "chroma", "milvus")

_EMBEDDING_KEY_TOKENS = ("embedding", "embed", "vector", "retriev")
_LLM_KEY_TOKENS = ("chat", "completion", "llm", "gpt")


def _classify_paired_model(model_id: str) -> str | None:
    """Use the 3-tier model registry to classify a paired model.

    Tiers (already cached, warm by the time endpoint_classifier runs):
      1. LiteLLM catalog (2500+ commercial models, 24h disk cache)
      2. Builtin regex patterns (~60 families including embedding models)
      3. HuggingFace Hub (1M+ models, per-model disk cache)

    Returns ``'embedding'``, ``'llm'``, or ``None`` if unrecognised.
    """
    meta = registry_lookup(model_id)
    if meta is None:
        return None
    family = meta.get("family", "")
    if family == "embedding" or "embed" in family.lower():
        return "embedding"
    pt = meta.get("pipeline_tag", "")
    if pt in ("feature-extraction", "sentence-similarity"):
        return "embedding"
    return "llm"


def classify_endpoint(
    url: str,
    context_key: str = "",
    paired_model: str = "",
) -> str:
    """Classify an endpoint URL.

    Returns one of ``'llm_endpoint'``, ``'model_endpoint'``, or
    ``'vector_store'``.
    """
    url_lower = url.lower()

    for vs in _VECTOR_STORE_URL_TOKENS:
        if vs in url_lower:
            return "vector_store"

    for _provider, pattern in _CLOUD_LLM_PROVIDER_PATTERNS:
        if pattern.search(url):
            return "llm_endpoint"

    if paired_model:
        kind = _classify_paired_model(paired_model)
        if kind == "embedding":
            return "model_endpoint"
        if kind == "llm":
            return "llm_endpoint"

    if context_key:
        key_lower = context_key.lower()
        if any(tok in key_lower for tok in _EMBEDDING_KEY_TOKENS):
            return "model_endpoint"
        if any(tok in key_lower for tok in _LLM_KEY_TOKENS):
            return "llm_endpoint"

    return "model_endpoint"
