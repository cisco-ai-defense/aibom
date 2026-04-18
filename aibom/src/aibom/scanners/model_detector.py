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

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import quote_plus

import yaml
from pathspec import PathSpec

from ..cache_paths import cache_read_dirs, ensure_cache_dir
from .file_cache import is_python_source, read_python_source, read_text_cached

from ..models import AIComponent, ComponentRelationship, ScanContext
from ..models.enums import AIComponentType, DetectionSource
from .base import BaseScanner

_LOGGER = logging.getLogger(__name__)

_MODEL_CATALOG_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
_HF_API_URL = "https://huggingface.co/api/models"
_CACHE_TTL_SECONDS = 86400  # 24 hours

_HF_SLUG_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

# ---------------------------------------------------------------------------
# Generic cache helpers
# ---------------------------------------------------------------------------


def _cache_file(name: str) -> Path:
    cache_dir = ensure_cache_dir("model")
    return cache_dir / name


def _load_cached(name: str) -> dict[str, dict[str, Any]] | None:
    for cache_dir in cache_read_dirs("model"):
        cp = cache_dir / name
        if not cp.exists():
            continue
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
            if time.time() - data.get("_ts", 0) > _CACHE_TTL_SECONDS:
                return None
            return data.get("models", {})
        except Exception:
            return None
    return None


def _save_cached(name: str, models: dict[str, dict[str, Any]]) -> None:
    try:
        cp = _cache_file(name)
        cp.write_text(
            json.dumps({"_ts": time.time(), "models": models}),
            encoding="utf-8",
        )
    except Exception:
        _LOGGER.debug("Failed to write cache %s", name, exc_info=True)


# ---------------------------------------------------------------------------
# Tier 1 -- Model Catalog  (commercial API models, 2500+)
# ---------------------------------------------------------------------------


def _fetch_model_catalog(
    provider: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch the LiteLLM model catalog and normalize it to a list of dicts.

    The canonical source is the GitHub-hosted ``model_prices_and_context_window.json``
    which is a ``dict[model_id, metadata]`` payload. We also tolerate the
    legacy ``{"data": [...]}`` / ``{"models": [...]}`` shapes and a bare list
    in case an alternate mirror is substituted.
    """
    import httpx

    params: dict[str, str] = {}
    if provider:
        params["provider"] = provider
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(_MODEL_CATALOG_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            if "data" in data or "models" in data:
                entries = data.get("data", data.get("models", []))
            else:
                entries = [
                    {"model_name": k, **v}
                    for k, v in data.items()
                    if k != "sample_spec" and isinstance(v, dict)
                ]
        else:
            entries = []

        if not entries:
            _LOGGER.warning(
                "Model catalog fetch returned 0 entries from %s",
                _MODEL_CATALOG_URL,
            )
        return entries
    except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
        _LOGGER.warning(
            "Model catalog fetch failed from %s: %s",
            _MODEL_CATALOG_URL,
            exc,
        )
    return []


def _provider_from_model_id(model_id: str) -> str:
    if "/" in model_id:
        return model_id.split("/", 1)[0]
    return "unknown"


_BEDROCK_VERSION_RE = re.compile(r":[0-9]+$")
_DATE_SUFFIX_RE = re.compile(r"-\d{4}-?\d{2}-?\d{2}(-v\d+)?$")


def _model_alias_keys(raw_id: str) -> list[str]:
    """Generate all normalized lookup keys for a model catalog ID.

    Handles: slash-prefixed (azure/gpt-4o), dot-prefixed Bedrock ARNs
    (us.anthropic.claude-3-5-sonnet-20241022-v2:0), fine-tune prefixes
    (ft:gpt-4o-2024-08-06), and date suffixes (-20241022).
    """
    keys: list[str] = []
    low = raw_id.lower()
    keys.append(low)

    current = low

    # Strip slash prefix: azure/eu/gpt-4o-2024-08-06 → gpt-4o-2024-08-06
    if "/" in current:
        current = current.rsplit("/", 1)[-1]
        keys.append(current)

    # Strip ft: prefix: ft:gpt-4o-2024-08-06 → gpt-4o-2024-08-06
    if current.startswith("ft:"):
        current = current[3:]
        keys.append(current)

    # Strip dot-separated provider/region prefix for Bedrock ARNs:
    #   us.anthropic.claude-3-5-sonnet-20241022-v2:0
    #   → anthropic.claude-3-5-sonnet-20241022-v2:0
    #   → claude-3-5-sonnet-20241022-v2:0
    if "." in current:
        parts = current.split(".")
        stripped = current
        for part in parts[:-1]:
            stripped = stripped[len(part) + 1 :]
            if stripped:
                keys.append(stripped)
        current = parts[-1]

    # Strip Bedrock version suffix: claude-3-5-sonnet-20241022-v2:0 → claude-3-5-sonnet-20241022-v2
    no_ver = _BEDROCK_VERSION_RE.sub("", current)
    if no_ver != current:
        keys.append(no_ver)
        current = no_ver

    # Strip date suffix: claude-3-5-sonnet-20241022-v2 → claude-3-5-sonnet
    #                     gpt-4o-2024-08-06 → gpt-4o
    no_date = _DATE_SUFFIX_RE.sub("", current)
    if no_date != current and len(no_date) >= 3:
        keys.append(no_date)

    return list(dict.fromkeys(keys))


_NATIVE_PROVIDERS = frozenset(
    {
        "openai",
        "anthropic",
        "google",
        "gemini",
        "mistral",
        "cohere",
        "deepseek",
        "xai",
        "perplexity",
        "groq",
        "fireworks_ai",
        "together_ai",
        "ollama",
        "huggingface",
        "replicate",
        "ai21",
        "writer",
        "voyage",
    }
)

# LiteLLM sometimes uses internal provider aliases that differ from how
# users think of the provider (e.g. ``gemini`` is Google's native API,
# ``bedrock_converse`` is still AWS Bedrock). Canonicalize to the names
# AIBOM's built-in registry and documentation use so the output is
# consistent across resolution paths.
_PROVIDER_CANONICAL_MAP: dict[str, str] = {
    "gemini": "google",
    "bedrock_converse": "bedrock",
    "vertex_ai-anthropic_models": "vertex_ai",
    "vertex_ai-gemini_models": "vertex_ai",
    "vertex_ai-ai21_models": "vertex_ai",
    "vertex_ai-code-chat-models": "vertex_ai",
    "vertex_ai-chat-models": "vertex_ai",
    "vertex_ai-code-text-models": "vertex_ai",
    "vertex_ai-text-models": "vertex_ai",
    "vertex_ai-language-models": "vertex_ai",
    "vertex_ai-image-models": "vertex_ai",
    "vertex_ai-embedding-models": "vertex_ai",
    "vertex_ai-mistral_models": "vertex_ai",
    "vertex_ai-llama_models": "vertex_ai",
    "bedrock/invoke": "bedrock",
}


def _canonical_provider(provider: str) -> str:
    low = provider.lower()
    return _PROVIDER_CANONICAL_MAP.get(low, low)


def _is_native_provider(provider: str) -> bool:
    canonical = _canonical_provider(provider)
    return canonical in _NATIVE_PROVIDERS


def _build_model_registry() -> dict[str, dict[str, Any]]:
    cached = _load_cached("model_catalog.json")
    if cached is not None:
        return cached

    raw = _fetch_model_catalog()
    if not raw:
        return {}

    # Four-tier registration so canonical (alias == literal model_id) and
    # native-provider entries win regardless of the catalog's iteration
    # order. Without this, the LiteLLM catalog's alphabetical ordering
    # causes e.g. ``azure/gpt-4o-2024-08-06`` (alias ``gpt-4o``) to hijack
    # the provider attribution of the native OpenAI ``gpt-4o`` entry.
    registry: dict[str, dict[str, Any]] = {}
    prepared: list[tuple[str, list[str], dict[str, Any], bool]] = []
    for entry in raw:
        model_id = entry.get("model_name") or entry.get("id") or entry.get("model") or ""
        if not model_id:
            continue
        provider = (
            entry.get("provider")
            or entry.get("litellm_provider")
            or _provider_from_model_id(model_id)
        )
        aliases = _model_alias_keys(model_id)
        canonical = aliases[-1] if aliases else model_id.lower()
        meta = {
            "provider": _canonical_provider(provider),
            "family": canonical.rsplit("-", 1)[0] if "-" in canonical else canonical,
            "mode": str(entry.get("mode") or "").lower(),
            "deprecated": entry.get(
                "deprecated", bool(entry.get("deprecation_date"))
            ),
            "source": "model_catalog",
        }
        prepared.append(
            (model_id.lower(), aliases, meta, _is_native_provider(provider))
        )

    def _register(require_canonical: bool, require_native: bool) -> None:
        for mid_low, aliases, meta, is_native in prepared:
            if require_native and not is_native:
                continue
            if not require_native and is_native:
                continue
            for alias in aliases:
                if require_canonical and alias != mid_low:
                    continue
                if alias not in registry:
                    registry[alias] = meta

    # Pass order: canonical-native, canonical-gateway, alias-native, alias-gateway.
    _register(require_canonical=True, require_native=True)
    _register(require_canonical=True, require_native=False)
    _register(require_canonical=False, require_native=True)
    _register(require_canonical=False, require_native=False)

    _save_cached("model_catalog.json", registry)
    return registry


_model_registry: dict[str, dict[str, Any]] | None = None


def _get_live_registry() -> dict[str, dict[str, Any]]:
    global _model_registry
    if _model_registry is None:
        _model_registry = _build_model_registry()
    return _model_registry


# ---------------------------------------------------------------------------
# Tier 2 -- HuggingFace Hub  (open-weight models, 1M+)
# ---------------------------------------------------------------------------

_hf_cache: dict[str, dict[str, Any] | None] = {}


def _is_hf_model_id(model_id: str) -> bool:
    """Return True if model_id looks like a HuggingFace Hub slug (org/name)."""
    return bool(_HF_SLUG_RE.match(model_id))


def _query_hf_hub(model_id: str) -> dict[str, Any] | None:
    """Look up a single model on HuggingFace Hub. Returns metadata or None."""
    if model_id in _hf_cache:
        return _hf_cache[model_id]

    cache_name = "hf_hub_models.json"
    disk = _load_cached(cache_name)
    if disk and model_id in disk:
        _hf_cache[model_id] = disk[model_id]
        return disk[model_id]

    import httpx

    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(f"{_HF_API_URL}/{model_id}")
            if resp.status_code == 404:
                _hf_cache[model_id] = None
                return None
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        _LOGGER.debug("HuggingFace Hub lookup failed for %s", model_id, exc_info=True)
        _hf_cache[model_id] = None
        return None

    org = model_id.split("/", 1)[0]
    model_name_part = model_id.split("/", 1)[-1]
    tags = data.get("tags", [])
    pipeline_tag = data.get("pipeline_tag", "")

    result: dict[str, Any] = {
        "provider": org,
        "family": model_name_part.rsplit("-", 1)[0] if "-" in model_name_part else model_name_part,
        "deprecated": False,
        "source": "huggingface",
        "hf_id": model_id,
        "pipeline_tag": pipeline_tag,
        "license": data.get("cardData", {}).get("license", ""),
        "downloads": data.get("downloads", 0),
        "tags": tags[:10],
    }

    _hf_cache[model_id] = result
    existing = _load_cached(cache_name) or {}
    existing[model_id] = result
    _save_cached(cache_name, existing)
    return result


BUILTIN_MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    r"^gpt-4o-\d{4}-\d{2}-\d{2}$": {
        "provider": "openai",
        "family": "gpt-4o",
        "deprecated": False,
    },
    r"^gpt-4-\d{4}-\d{2}-\d{2}$": {
        "provider": "openai",
        "family": "gpt-4",
        "deprecated": False,
    },
    r"^gpt-3\.5-turbo-\d{4}-\d{2}-\d{2}$": {
        "provider": "openai",
        "family": "gpt-3.5-turbo",
        "deprecated": False,
    },
    r"^gpt-4o-mini$": {"provider": "openai", "family": "gpt-4o", "deprecated": False},
    r"^gpt-4o$": {"provider": "openai", "family": "gpt-4o", "deprecated": False},
    r"^gpt-4-turbo$": {"provider": "openai", "family": "gpt-4", "deprecated": False},
    r"^gpt-4$": {"provider": "openai", "family": "gpt-4", "deprecated": False},
    r"^gpt-4-32k": {"provider": "openai", "family": "gpt-4", "deprecated": True},
    r"^gpt-4\.1$": {"provider": "openai", "family": "gpt-4.1", "deprecated": False},
    r"^gpt-4\.1-mini$": {"provider": "openai", "family": "gpt-4.1", "deprecated": False},
    r"^gpt-4\.1-nano$": {"provider": "openai", "family": "gpt-4.1", "deprecated": False},
    r"^gpt-4\.5": {"provider": "openai", "family": "gpt-4.5", "deprecated": False},
    r"^gpt-3\.5-turbo$": {
        "provider": "openai",
        "family": "gpt-3.5",
        "deprecated": False,
    },
    r"^o1-pro$": {"provider": "openai", "family": "o1", "deprecated": False},
    r"^o1-mini$": {"provider": "openai", "family": "o1", "deprecated": False},
    r"^o1$": {"provider": "openai", "family": "o1", "deprecated": False},
    r"^o3-mini$": {"provider": "openai", "family": "o3", "deprecated": False},
    r"^o3$": {"provider": "openai", "family": "o3", "deprecated": False},
    r"^o4-mini$": {"provider": "openai", "family": "o4", "deprecated": False},
    r"^text-embedding-ada-002$": {"provider": "openai", "family": "embedding", "deprecated": True},
    r"^text-embedding-3-(small|large)$": {"provider": "openai", "family": "embedding", "deprecated": False},
    r"^claude-3-5-sonnet-\d{8}$": {
        "provider": "anthropic",
        "family": "claude-3-5-sonnet",
        "deprecated": False,
    },
    r"^claude-3-5-haiku-\d{8}$": {
        "provider": "anthropic",
        "family": "claude-3-5-haiku",
        "deprecated": False,
    },
    r"^claude-3-7-sonnet-\d{8}$": {
        "provider": "anthropic",
        "family": "claude-3-7-sonnet",
        "deprecated": False,
    },
    r"^claude-3-opus-\d{8}$": {
        "provider": "anthropic",
        "family": "claude-3-opus",
        "deprecated": False,
    },
    r"^claude-3-sonnet-\d{8}$": {
        "provider": "anthropic",
        "family": "claude-3-sonnet",
        "deprecated": False,
    },
    r"^claude-3-haiku-\d{8}$": {
        "provider": "anthropic",
        "family": "claude-3-haiku",
        "deprecated": False,
    },
    r"^claude-4-sonnet-\d{8}$": {
        "provider": "anthropic",
        "family": "claude-4-sonnet",
        "deprecated": False,
    },
    r"^claude-4-opus-\d{8}$": {
        "provider": "anthropic",
        "family": "claude-4-opus",
        "deprecated": False,
    },
    r"^claude-sonnet-4-\d{8}$": {
        "provider": "anthropic",
        "family": "claude-sonnet-4",
        "deprecated": False,
    },
    r"^claude-opus-4-\d{8}$": {
        "provider": "anthropic",
        "family": "claude-opus-4",
        "deprecated": False,
    },
    r"^claude-haiku-4-\d{8}$": {
        "provider": "anthropic",
        "family": "claude-haiku-4",
        "deprecated": False,
    },
    r"^claude-3-5-sonnet$": {
        "provider": "anthropic",
        "family": "claude-3-5-sonnet",
        "deprecated": False,
    },
    r"^claude-3-5-haiku$": {
        "provider": "anthropic",
        "family": "claude-3-5-haiku",
        "deprecated": False,
    },
    r"^claude-3-7-sonnet$": {
        "provider": "anthropic",
        "family": "claude-3-7-sonnet",
        "deprecated": False,
    },
    r"^claude-3-opus$": {
        "provider": "anthropic",
        "family": "claude-3-opus",
        "deprecated": False,
    },
    r"^claude-3-sonnet$": {
        "provider": "anthropic",
        "family": "claude-3-sonnet",
        "deprecated": False,
    },
    r"^claude-3-haiku$": {
        "provider": "anthropic",
        "family": "claude-3-haiku",
        "deprecated": False,
    },
    r"^claude-4-sonnet$": {
        "provider": "anthropic",
        "family": "claude-4-sonnet",
        "deprecated": False,
    },
    r"^claude-4-opus$": {
        "provider": "anthropic",
        "family": "claude-4-opus",
        "deprecated": False,
    },
    r"^claude-sonnet-4(?:-\d+)?$": {
        "provider": "anthropic",
        "family": "claude-sonnet-4",
        "deprecated": False,
    },
    r"^claude-opus-4(?:-\d+)?$": {
        "provider": "anthropic",
        "family": "claude-opus-4",
        "deprecated": False,
    },
    r"^claude-haiku-4(?:-\d+)?$": {
        "provider": "anthropic",
        "family": "claude-haiku-4",
        "deprecated": False,
    },
    r"^gemini-2\.5-pro$": {
        "provider": "google",
        "family": "gemini-2.5",
        "deprecated": False,
    },
    r"^gemini-2\.0-flash$": {
        "provider": "google",
        "family": "gemini-2.0",
        "deprecated": False,
    },
    r"^gemini-1\.5-flash$": {
        "provider": "google",
        "family": "gemini-1.5",
        "deprecated": False,
    },
    r"^gemini-1\.5-pro$": {
        "provider": "google",
        "family": "gemini-1.5",
        "deprecated": False,
    },
    r"^llama[-_]4": {"provider": "meta", "family": "llama-4", "deprecated": False},
    r"^llama[-_]3\.3": {"provider": "meta", "family": "llama-3.3", "deprecated": False},
    r"^llama[-_]3\.2": {"provider": "meta", "family": "llama-3.2", "deprecated": False},
    r"^llama[-_]3\.1": {"provider": "meta", "family": "llama-3.1", "deprecated": False},
    r"^llama[-_]3\b": {"provider": "meta", "family": "llama-3", "deprecated": False},
    r"^llama[-_]2\b": {"provider": "meta", "family": "llama-2", "deprecated": False},
    # Bedrock-style Llama IDs omit the separator between "llama" and the major
    # version, e.g. ``meta.llama3-70b-instruct-v1:0`` → alias ``llama3-70b-instruct-v1``.
    # The minor version is expressed with a hyphen: ``llama3-1-`` is v3.1, ``llama3-2-``
    # is v3.2, etc. Order matters: the more specific minor-version patterns must
    # come before the bare ``^llama3\b`` fallback.
    r"^llama4\b": {"provider": "meta", "family": "llama-4", "deprecated": False},
    r"^llama3-3\b": {"provider": "meta", "family": "llama-3.3", "deprecated": False},
    r"^llama3-2\b": {"provider": "meta", "family": "llama-3.2", "deprecated": False},
    r"^llama3-1\b": {"provider": "meta", "family": "llama-3.1", "deprecated": False},
    r"^llama3\b": {"provider": "meta", "family": "llama-3", "deprecated": False},
    r"^llama2\b": {"provider": "meta", "family": "llama-2", "deprecated": False},
    r"^mistral-7b": {
        "provider": "mistral",
        "family": "mistral-7b",
        "deprecated": False,
    },
    r"^mistral-large": {
        "provider": "mistral",
        "family": "mistral-large",
        "deprecated": False,
    },
    r"^mistral-medium": {
        "provider": "mistral",
        "family": "mistral-medium",
        "deprecated": False,
    },
    r"^mistral-small": {
        "provider": "mistral",
        "family": "mistral-small",
        "deprecated": False,
    },
    r"^mixtral": {"provider": "mistral", "family": "mixtral", "deprecated": False},
    r"^codestral": {"provider": "mistral", "family": "codestral", "deprecated": False},
    r"^pixtral": {"provider": "mistral", "family": "pixtral", "deprecated": False},
    r"^phi-4\b": {"provider": "microsoft", "family": "phi-4", "deprecated": False},
    r"^phi-3\b": {"provider": "microsoft", "family": "phi-3", "deprecated": False},
    r"^qwen2\.5": {"provider": "qwen", "family": "qwen2.5", "deprecated": False},
    r"^qwen2\b": {"provider": "qwen", "family": "qwen2", "deprecated": False},
    r"^qwen\b": {"provider": "qwen", "family": "qwen", "deprecated": False},
    r"^deepseek-r1": {
        "provider": "deepseek",
        "family": "deepseek-r1",
        "deprecated": False,
    },
    r"^deepseek-v3": {
        "provider": "deepseek",
        "family": "deepseek-v3",
        "deprecated": False,
    },
    r"^deepseek-v2": {
        "provider": "deepseek",
        "family": "deepseek-v2",
        "deprecated": False,
    },
    r"^command-r-plus(?:-v\d+)?$": {
        "provider": "cohere",
        "family": "command-r-plus",
        "deprecated": False,
    },
    r"^command-r(?:-v\d+)?$": {
        "provider": "cohere",
        "family": "command-r",
        "deprecated": False,
    },
    r"^command-light(?:-text)?(?:-v\d+)?$": {
        "provider": "cohere",
        "family": "command-light",
        "deprecated": False,
    },
    r"^command(?:-text)?(?:-v\d+)?$": {
        "provider": "cohere",
        "family": "command",
        "deprecated": False,
    },
    r"^embed-english(?:-light)?(?:-v\d+)?$": {
        "provider": "cohere",
        "family": "embed-english",
        "deprecated": False,
    },
    r"^embed-multilingual(?:-light)?(?:-v\d+)?$": {
        "provider": "cohere",
        "family": "embed-multilingual",
        "deprecated": False,
    },
    r"^nova-pro(?:-v\d+)?$": {
        "provider": "amazon",
        "family": "nova-pro",
        "deprecated": False,
    },
    r"^nova-lite(?:-v\d+)?$": {
        "provider": "amazon",
        "family": "nova-lite",
        "deprecated": False,
    },
    r"^nova-micro(?:-v\d+)?$": {
        "provider": "amazon",
        "family": "nova-micro",
        "deprecated": False,
    },
    r"^nova-canvas(?:-v\d+)?$": {
        "provider": "amazon",
        "family": "nova-canvas",
        "deprecated": False,
    },
    r"^nova-reel(?:-v\d+)?$": {
        "provider": "amazon",
        "family": "nova-reel",
        "deprecated": False,
    },
    r"^nova-sonic(?:-v\d+)?$": {
        "provider": "amazon",
        "family": "nova-sonic",
        "deprecated": False,
    },
    r"^titan-text-(?:express|lite|premier|agile)(?:-v\d+)?$": {
        "provider": "amazon",
        "family": "titan-text",
        "deprecated": False,
    },
    r"^titan-embed-text(?:-v\d+)?$": {
        "provider": "amazon",
        "family": "titan-embed-text",
        "deprecated": False,
    },
    r"^titan-embed-image(?:-v\d+)?$": {
        "provider": "amazon",
        "family": "titan-embed-image",
        "deprecated": False,
    },
    r"^titan-image-generator(?:-v\d+)?$": {
        "provider": "amazon",
        "family": "titan-image-generator",
        "deprecated": False,
    },
}

_COMPILED_REGISTRY: list[tuple[re.Pattern[str], dict[str, Any]]] = [
    (re.compile(pat, re.IGNORECASE), meta)
    for pat, meta in BUILTIN_MODEL_REGISTRY.items()
]

_PY_KWARG_RE = re.compile(
    r"\b(?:model|model_name|deployment_name|model_id)\s*=\s*"
    r"(?P<q>[\"'])(?P<val>[^\"'\\]*(?:\\.[^\"'\\]*)*)(?P=q)",
    re.MULTILINE,
)

# Module-level assignments whose *target* name is model-related.
#
# Two alternatives:
#   1. Lowercase exact: ``model``, ``model_name``, ``model_id``, ``deployment_name``
#   2. UPPER_SNAKE with a model suffix, optionally preceded by a namespace
#      prefix (e.g. ``MODEL_ID``, ``BEDROCK_MODEL_ID``, ``CLAUDE_MODEL``,
#      ``SONNET_MODEL_NAME``, ``OPENAI_DEPLOYMENT``, ``LLM_MODEL``).
#
# The value must be a quoted literal. Non-model UPPER_SNAKE vars (e.g.
# ``LOG_LEVEL`` or ``API_URL``) do not match because they lack the suffix.
_PY_ASSIGN_RE = re.compile(
    r"(?m)^\s*"
    r"(?:"
    r"model|model_name|model_id|deployment_name"
    r"|"
    r"(?:[A-Z][A-Z0-9_]*_)?(?:MODEL|MODEL_ID|MODEL_NAME|DEPLOYMENT|DEPLOYMENT_NAME|LLM_MODEL)"
    r")"
    r"\s*=\s*"
    r"(?P<q>[\"'])(?P<val>[^\"'\\]*(?:\\.[^\"'\\]*)*)(?P=q)",
)

# ``os.getenv(...)`` / ``os.environ.get(...)`` with a quoted default value
# whose env var name looks model-related. Strands and similar agent
# frameworks commonly read the deployed model ID from the environment with a
# hard-coded literal fallback — that literal is the real model in practice.
#
# We constrain the env var name to model-suffixed identifiers so that calls
# like ``os.getenv("LOG_LEVEL", "info")`` do not falsely register "info" as
# a model ID.
_PY_GETENV_WITH_DEFAULT_RE = re.compile(
    r"(?:os\.)?(?:getenv|environ\.get)\s*\(\s*"
    r"(?P<envq>[\"'])"
    r"(?P<env>(?:[A-Z][A-Z0-9_]*_)?(?:MODEL|MODEL_ID|MODEL_NAME|DEPLOYMENT|DEPLOYMENT_NAME|LLM_MODEL))"
    r"(?P=envq)\s*,\s*"
    r"(?P<q>[\"'])(?P<val>[^\"'\\]*(?:\\.[^\"'\\]*)*)(?P=q)",
)

_PY_CTOR_RE = re.compile(
    r"\b(?:ChatOpenAI|ChatAnthropic|ChatGoogleGenerativeAI|AzureChatOpenAI|"
    r"OpenAI|Anthropic|GenerativeModel)\s*\("
    r"[^)]*?\bmodel\s*=\s*(?P<q>[\"'])(?P<val>[^\"'\\]*(?:\\.[^\"'\\]*)*)(?P=q)",
    re.DOTALL,
)

# Subscript assignments like ``agent.model.config['model_id'] = 'claude-3-5-...'``
# or ``config["model_name"] = "gpt-4o"``. Strands and a few other agent
# frameworks expose the active model through a ``config`` dict on the model
# object, and the pattern is common enough in tutorials and tests that
# missing it leaves real model identifiers out of the BOM.
_PY_SUBSCRIPT_MODEL_RE = re.compile(
    r"(?m)"
    r"(?:[A-Za-z_][\w.]*)?"
    r"\s*\[\s*(?P<kq>[\"'])"
    r"(?P<key>model|model_id|model_name|deployment_name)"
    r"(?P=kq)\s*\]"
    r"\s*=\s*"
    r"(?P<q>[\"'])(?P<val>[^\"'\\]*(?:\\.[^\"'\\]*)*)(?P=q)",
)

_ENV_MODEL_RE = re.compile(
    r"(?m)^\s*(?:MODEL|OPENAI_MODEL|LLM_MODEL|ANTHROPIC_MODEL)\s*=\s*"
    r"(?:[\"']([^\"']+)[\"']|([^\s#]+))",
)

_TOML_HEADER_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")
_TOML_KV_RE = re.compile(
    r"^\s*(?:model|model_name|model_id|deployment_name)\s*=\s*"
    r"(?:[\"']([^\"']+)[\"']|([^\s#]+))",
)

_JSON_KEY_HINT = frozenset({"model", "model_name", "model_id", "deployment_name"})

_SKIP_STRINGS = frozenset(
    {
        "",
        "true",
        "false",
        "null",
        "none",
        "auto",
        "default",
    }
)


def _model_card_url(model_name: str, provider: str, meta: Optional[dict[str, Any]] = None) -> str:
    if meta and meta.get("source") == "huggingface" and meta.get("hf_id"):
        return f"https://huggingface.co/{meta['hf_id']}"
    p = provider.lower()
    if p in ("meta", "mistral", "qwen", "deepseek", "microsoft"):
        return f"https://huggingface.co/models?search={quote_plus(model_name)}"
    if p == "openai":
        return "https://platform.openai.com/docs/models"
    if p == "anthropic":
        return "https://docs.anthropic.com/en/docs/about-claude/models"
    if p == "google":
        return "https://ai.google.dev/gemini-api/docs/models/gemini"
    if p == "cohere":
        return "https://docs.cohere.com/docs/models"
    return f"https://huggingface.co/models?search={quote_plus(model_name)}"


def _line_number(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _normalize_candidate(raw: str) -> str:
    s = raw.strip().strip('"').strip("'")
    if s.startswith("${") or "{{" in s:
        return ""
    return s


def _is_plausible_model_id(s: str) -> bool:
    if len(s) < 2 or len(s) > 256:
        return False
    low = s.lower()
    if low in _SKIP_STRINGS or low.startswith("${"):
        return False
    if not re.search(r"[a-zA-Z]", s):
        return False
    return True


_registry_cache: dict[str, dict[str, Any] | None] = {}
_SENTINEL = object()


def _builtin_regex_match_direct(key: str) -> Optional[dict[str, Any]]:
    """Return the first BUILTIN_MODEL_REGISTRY match for ``key`` itself (no alias walk)."""
    low = key.lower()
    for pat, meta in _COMPILED_REGISTRY:
        if pat.search(low):
            return dict(meta)
    return None


def _builtin_regex_lookup(key: str) -> Optional[dict[str, Any]]:
    """Return the first BUILTIN_MODEL_REGISTRY match for ``key`` or any alias.

    HuggingFace-style ``org/name`` slugs are resolved via Tier 3 (HF Hub)
    instead of alias-walking into this registry, which would incorrectly
    match e.g. ``meta-llama/Llama-3.1-8B-Instruct`` against the generic
    ``^llama[-_]3\\.1`` pattern and mask HF provenance.
    """
    direct = _builtin_regex_match_direct(key)
    if direct:
        return direct
    if _is_hf_model_id(key):
        return None
    for alias in _model_alias_keys(key):
        if alias == key.lower():
            continue
        for pat, meta in _COMPILED_REGISTRY:
            if pat.search(alias):
                return dict(meta)
    return None


def is_known_embedding_model_name(name: str) -> bool:
    """Return True iff ``name`` is a known embedding model identifier.

    Uses :func:`registry_lookup` as the single source of truth — which
    consults the live LiteLLM catalog (where ``mode == "embedding"`` is
    authoritative for thousands of entries), then the hand-curated
    built-in registry (where ``family == "embedding"`` is set for the
    well-known OpenAI ``text-embedding-*`` families), then HuggingFace.
    Unknown strings return False.
    """
    if not name:
        return False
    cleaned = name.strip()
    if not cleaned:
        return False
    meta = registry_lookup(cleaned)
    if meta is None:
        return False
    if str(meta.get("mode", "")).lower() == "embedding":
        return True
    if str(meta.get("family", "")).lower().startswith("embed"):
        return True
    return False


def registry_lookup(model_id: str) -> Optional[dict[str, Any]]:
    key = model_id.strip()
    cached = _registry_cache.get(key, _SENTINEL)
    if cached is not _SENTINEL:
        return dict(cached) if cached is not None else None

    # Tier 1: LiteLLM commercial API catalog. Walk alias forms (e.g.
    # ``us.anthropic.claude-3-7-sonnet-...`` → ``claude-3-7-sonnet``) so
    # inference-profile prefixes resolve against the flattened registry.
    live = _get_live_registry()
    live_hit: Optional[dict[str, Any]] = None
    for candidate in [key.lower()] + [
        a for a in _model_alias_keys(key) if a != key.lower()
    ]:
        hit = live.get(candidate)
        if hit:
            live_hit = hit
            break

    # Gateway override: when the input is a native shorthand (e.g.
    # ``claude-3-5-sonnet``) the LiteLLM catalog may only know it via a
    # cloud-gateway alias row (bedrock, vertex_ai, vercel_ai_gateway). Prefer
    # the curated built-in regex, which has accurate native-provider
    # attribution. We only apply this when the built-in regex matches the
    # INPUT directly — otherwise a real Bedrock-formatted ID like
    # ``anthropic.claude-3-5-sonnet-20241022-v2:0`` (whose alias walk also
    # hits the builtin regex) would be misclassified as native ``anthropic``.
    if live_hit and not _is_native_provider(live_hit.get("provider", "")):
        direct_builtin = _builtin_regex_match_direct(key)
        if direct_builtin and _is_native_provider(
            direct_builtin.get("provider", "")
        ):
            result = dict(direct_builtin)
            _registry_cache[key] = result
            return result

    if live_hit:
        result = dict(live_hit)
        _registry_cache[key] = result
        return result

    # Tier 2 (offline): built-in regex patterns, including alias walk for
    # inputs like ``us.anthropic.claude-3-7-sonnet-...`` when the live catalog
    # is unavailable.
    builtin_hit = _builtin_regex_lookup(key)
    if builtin_hit:
        _registry_cache[key] = builtin_hit
        return dict(builtin_hit)

    # Tier 3: HuggingFace Hub for org/model-name slugs
    if _is_hf_model_id(key):
        hf = _query_hf_hub(key)
        if hf:
            result = dict(hf)
            _registry_cache[key] = result
            return result

    _registry_cache[key] = None
    return None


def _build_pathspec(patterns: list[str]) -> Optional[PathSpec]:
    if not patterns:
        return None
    return PathSpec.from_lines("gitwildmatch", patterns)


def _iter_files(context: ScanContext) -> Iterator[tuple[Path, str]]:
    idx = context.file_index()
    if idx:
        for entries in idx.values():
            for entry in entries:
                try:
                    rel = entry.path.relative_to(entry.root).as_posix()
                except ValueError:
                    rel = entry.path.as_posix()
                yield entry.path, rel
        return

    spec = _build_pathspec(context.exclude_patterns)
    for scan_root in context.paths:
        root = Path(scan_root)
        if not root.exists():
            continue
        if root.is_file():
            rel = root.name
            if spec and spec.match_file(rel):
                continue
            yield root, rel
            continue
        base = root.resolve()
        for f in root.rglob("*"):
            if not f.is_file():
                continue
            try:
                rel = f.resolve().relative_to(base).as_posix()
            except ValueError:
                rel = f.as_posix()
            if spec and spec.match_file(rel):
                continue
            yield f, rel


def _line_for_value(text: str, value: str) -> int:
    if not value:
        return 0
    idx = text.find(value)
    if idx < 0:
        return 0
    return _line_number(text, idx)


def _walk_mapping(
    obj: Any,
    text: str,
    out: list[tuple[str, int]],
    depth: int,
) -> None:
    if depth > 64:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if (
                lk in _JSON_KEY_HINT
                and isinstance(v, str)
                and _is_plausible_model_id(v)
            ):
                out.append((v, _line_for_value(text, v)))
            _walk_mapping(v, text, out, depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            _walk_mapping(item, text, out, depth + 1)


def _extract_yaml_models(text: str) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return found
    if data is None:
        return found
    _walk_mapping(data, text, found, 0)
    return found


def _extract_json_models(text: str) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return found
    _walk_mapping(data, text, found, 0)
    return found


def _extract_toml_tool_models(text: str) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    in_tool = False
    for line_no, line in enumerate(text.splitlines(), start=1):
        m = _TOML_HEADER_RE.match(line)
        if m:
            in_tool = m.group(1).strip().lower().startswith("tool.")
            continue
        if not in_tool:
            continue
        km = _TOML_KV_RE.match(line)
        if km:
            val = km.group(1) or km.group(2) or ""
            if _is_plausible_model_id(val):
                found.append((val, line_no))
    return found


def _extract_env_models(text: str) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for m in _ENV_MODEL_RE.finditer(text):
        val = m.group(1) or m.group(2) or ""
        val = _normalize_candidate(val)
        if _is_plausible_model_id(val):
            found.append((val, _line_number(text, m.start())))
    return found


def _extract_python_models(text: str) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for rx in (
        _PY_KWARG_RE,
        _PY_ASSIGN_RE,
        _PY_CTOR_RE,
        _PY_GETENV_WITH_DEFAULT_RE,
        _PY_SUBSCRIPT_MODEL_RE,
    ):
        for m in rx.finditer(text):
            raw = m.group("val")
            s = _normalize_candidate(raw)
            if not _is_plausible_model_id(s):
                continue
            found.append((s, _line_number(text, m.start())))
    return found


def _make_component(
    model_name: str,
    file_path: str,
    line_number: int,
    method: str,
    meta: Optional[dict[str, Any]],
) -> AIComponent:
    if meta:
        provider = str(meta["provider"])
        family = str(meta["family"])
        deprecated = bool(meta.get("deprecated", False))
        confidence = 1.0
        url = _model_card_url(model_name, provider, meta)
        source = meta.get("source", "builtin")
        md: dict[str, Any] = {
            "model_card_url": url,
            "provider": provider,
            "family": family,
            "deprecated": deprecated,
            "detection_method": method,
            "registry_source": source,
        }
        if source == "huggingface":
            for hf_key in ("hf_id", "pipeline_tag", "license", "downloads"):
                if meta.get(hf_key):
                    md[hf_key] = meta[hf_key]
    else:
        provider = "unknown"
        confidence = 0.4
        url = _model_card_url(model_name, provider)
        md = {
            "model_card_url": url,
            "provider": provider,
            "family": "unknown",
            "deprecated": False,
            "detection_method": method,
            "registry_source": "none",
        }
    src = (
        DetectionSource.CONFIG_FILE
        if method in ("config_file", "env_var")
        else DetectionSource.CODE_ANALYSIS
    )
    needs_agentic = meta is None
    return AIComponent(
        name=model_name,
        component_type=AIComponentType.MODEL,
        file_path=file_path,
        line_number=line_number,
        framework=provider,
        detection_source=src,
        heuristic_confidence=confidence,
        needs_agentic=needs_agentic,
        agentic_hint=f"'{model_name}' not found in model registry; may be private or misidentified" if needs_agentic else "",
        model_name=model_name,
        metadata=md,
    )


class ModelDetector(BaseScanner):
    name = "model_detector"

    def supports(self, context: ScanContext) -> bool:
        return any(Path(p).exists() for p in context.paths)

    def scan(
        self, context: ScanContext
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        components: list[AIComponent] = []
        seen: set[tuple[str, str]] = set()

        for fpath, _rel in _iter_files(context):
            suffix = fpath.suffix.lower()
            name_lower = fpath.name.lower()
            try:
                text = (
                    read_python_source(fpath)
                    if is_python_source(fpath)
                    else read_text_cached(fpath)
                )
            except OSError:
                continue

            extracted: list[tuple[str, int, str]] = []

            if suffix in (".py", ".ipynb"):
                for val, ln in _extract_python_models(text):
                    extracted.append((val, ln, "string_literal"))
            elif name_lower == ".env" or fpath.name.endswith(".env"):
                for val, ln in _extract_env_models(text):
                    extracted.append((val, ln, "env_var"))
            elif suffix in (".yaml", ".yml"):
                for val, ln in _extract_yaml_models(text):
                    extracted.append((val, ln, "config_file"))
            elif suffix == ".json":
                for val, ln in _extract_json_models(text):
                    extracted.append((val, ln, "config_file"))
            elif suffix == ".toml":
                for val, ln in _extract_toml_tool_models(text):
                    extracted.append((val, ln, "config_file"))

            fp_str = str(fpath)
            for model_name, line_no, method in extracted:
                key = (fp_str, model_name)
                if key in seen:
                    continue
                seen.add(key)
                reg = registry_lookup(model_name)
                components.append(
                    _make_component(model_name, fp_str, line_no, method, reg)
                )

        return components, []
