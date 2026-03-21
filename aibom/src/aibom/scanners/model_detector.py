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
from platformdirs import user_cache_dir

from ..models import AIComponent, ComponentRelationship, ScanContext
from ..models.enums import AIComponentType, DetectionSource
from .base import BaseScanner

_LOGGER = logging.getLogger(__name__)

_LITELLM_CATALOG_URL = "https://models.litellm.ai/model_catalog"
_HF_API_URL = "https://huggingface.co/api/models"
_CACHE_TTL_SECONDS = 86400  # 24 hours
_CACHE_DIR = Path(user_cache_dir("aibom")) / "model_cache"

_HF_SLUG_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

# ---------------------------------------------------------------------------
# Generic cache helpers
# ---------------------------------------------------------------------------


def _cache_file(name: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / name


def _load_cached(name: str) -> dict[str, dict[str, Any]] | None:
    cp = _cache_file(name)
    if not cp.exists():
        return None
    try:
        data = json.loads(cp.read_text(encoding="utf-8"))
        if time.time() - data.get("_ts", 0) > _CACHE_TTL_SECONDS:
            return None
        return data.get("models", {})
    except Exception:
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
# Tier 1 -- LiteLLM Model Catalog  (commercial API models, 2500+)
# ---------------------------------------------------------------------------


def _fetch_litellm_catalog(
    provider: str | None = None,
) -> list[dict[str, Any]]:
    import httpx

    params: dict[str, str] = {}
    if provider:
        params["provider"] = provider
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(_LITELLM_CATALOG_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("data", data.get("models", []))
    except Exception:
        _LOGGER.debug("LiteLLM catalog fetch failed", exc_info=True)
    return []


def _provider_from_litellm_id(model_id: str) -> str:
    if "/" in model_id:
        return model_id.split("/", 1)[0]
    return "unknown"


_BEDROCK_VERSION_RE = re.compile(r":[0-9]+$")
_DATE_SUFFIX_RE = re.compile(r"-\d{4}-?\d{2}-?\d{2}(-v\d+)?$")


def _litellm_alias_keys(raw_id: str) -> list[str]:
    """Generate all normalized lookup keys for a LiteLLM model ID.

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


def _build_litellm_registry() -> dict[str, dict[str, Any]]:
    cached = _load_cached("litellm_models.json")
    if cached is not None:
        return cached

    raw = _fetch_litellm_catalog()
    if not raw:
        return {}

    registry: dict[str, dict[str, Any]] = {}
    for entry in raw:
        model_id = entry.get("model_name") or entry.get("id") or entry.get("model") or ""
        if not model_id:
            continue
        provider = (
            entry.get("provider")
            or entry.get("litellm_provider")
            or _provider_from_litellm_id(model_id)
        )
        aliases = _litellm_alias_keys(model_id)
        canonical = aliases[-1] if aliases else model_id.lower()
        meta = {
            "provider": provider.lower(),
            "family": canonical.rsplit("-", 1)[0] if "-" in canonical else canonical,
            "deprecated": entry.get("deprecated", False),
            "source": "litellm",
        }
        for alias in aliases:
            if alias not in registry:
                registry[alias] = meta
    _save_cached("litellm_models.json", registry)
    return registry


_litellm_registry: dict[str, dict[str, Any]] | None = None


def _get_live_registry() -> dict[str, dict[str, Any]]:
    global _litellm_registry
    if _litellm_registry is None:
        _litellm_registry = _build_litellm_registry()
    return _litellm_registry


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
    r"^command-r-plus$": {
        "provider": "cohere",
        "family": "command-r-plus",
        "deprecated": False,
    },
    r"^command-r$": {"provider": "cohere", "family": "command-r", "deprecated": False},
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

_PY_ASSIGN_RE = re.compile(
    r"(?m)^\s*(?:model|model_name|MODEL|MODEL_NAME|LLM_MODEL)\s*=\s*"
    r"(?P<q>[\"'])(?P<val>[^\"'\\]*(?:\\.[^\"'\\]*)*)(?P=q)",
)

_PY_CTOR_RE = re.compile(
    r"\b(?:ChatOpenAI|ChatAnthropic|ChatGoogleGenerativeAI|AzureChatOpenAI|"
    r"OpenAI|Anthropic|GenerativeModel)\s*\("
    r"[^)]*?\bmodel\s*=\s*(?P<q>[\"'])(?P<val>[^\"'\\]*(?:\\.[^\"'\\]*)*)(?P=q)",
    re.DOTALL,
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


def _registry_lookup(model_id: str) -> Optional[dict[str, Any]]:
    key = model_id.strip()

    # Tier 1: LiteLLM commercial API catalog
    live = _get_live_registry()
    hit = live.get(key.lower())
    if hit:
        return dict(hit)

    # Tier 2 (offline): builtin regex patterns
    for pat, meta in _COMPILED_REGISTRY:
        if pat.search(key):
            return dict(meta)

    # Tier 3: HuggingFace Hub for org/model-name slugs
    if _is_hf_model_id(key):
        hf = _query_hf_hub(key)
        if hf:
            return dict(hf)

    return None


def _build_pathspec(patterns: list[str]) -> Optional[PathSpec]:
    if not patterns:
        return None
    return PathSpec.from_lines("gitwildmatch", patterns)


def _iter_files(context: ScanContext) -> Iterator[tuple[Path, str]]:
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
    for rx in (_PY_KWARG_RE, _PY_ASSIGN_RE, _PY_CTOR_RE):
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
        confidence = 0.7
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
    return AIComponent(
        name=model_name,
        component_type=AIComponentType.MODEL,
        file_path=file_path,
        line_number=line_number,
        framework=provider,
        detection_source=src,
        confidence=confidence,
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
        seen: set[tuple[str, int, str, str]] = set()

        for fpath, _rel in _iter_files(context):
            suffix = fpath.suffix.lower()
            name_lower = fpath.name.lower()
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            extracted: list[tuple[str, int, str]] = []

            if suffix == ".py":
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
                key = (fp_str, line_no, model_name, method)
                if key in seen:
                    continue
                seen.add(key)
                reg = _registry_lookup(model_name)
                components.append(
                    _make_component(model_name, fp_str, line_no, method, reg)
                )

        return components, []
