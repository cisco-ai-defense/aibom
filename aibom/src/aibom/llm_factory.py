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

"""Centralised LLM construction.

Every code path that needs a ``BaseChatModel`` should call
:func:`build_chat_model` instead of duplicating provider-routing logic.
"""
from __future__ import annotations

import importlib
import logging
import re
from typing import Any

_logger = logging.getLogger(__name__)

# OpenAI reasoning-class models reject ``max_tokens`` (they require
# ``max_completion_tokens``) and reject a non-default ``temperature``. Matches
# the o-series (o1, o3, o4, …) and gpt-5.x reasoning families by the leaf model
# id (after any ``provider/`` prefix and any date/version suffix).
_REASONING_MODEL_RE = re.compile(
    r"^(?:o\d+(?:-|$)|gpt-5)",
    re.IGNORECASE,
)

# Newer-generation Anthropic Claude models — reached via the native
# ``anthropic`` provider OR via ``bedrock`` — removed the sampling parameters:
# sending an explicit ``temperature`` returns HTTP 400 (native) or
# ``ValidationException: temperature is deprecated for this model`` (Bedrock).
# The deprecation begins at Opus 4.7. Matched on the leaf id after
# stripping any ``provider/`` prefix, Bedrock regional inference-profile prefix
# (``us.``/``eu.``/``apac.``/``global.``), and the ``anthropic.`` vendor segment.
_TEMPERATURE_DEPRECATED_CLAUDE_RE = re.compile(
    r"^claude-(?:"
    r"opus-4-(?:[7-9]|\d{2,})"  # opus 4.7, 4.8, 4.9, 4.10+
    r"|opus-[5-9]"  # opus 5+
    r"|sonnet-[5-9]"  # sonnet 5+
    r"|fable-[5-9]"  # fable 5+
    r")",
    re.IGNORECASE,
)

_BEDROCK_REGION_PREFIX_RE = re.compile(
    r"^(?:us|eu|apac|global)\.", re.IGNORECASE
)


def _leaf_model_id(model_id: str) -> str:
    """Normalize *model_id* to a bare leaf for capability regexes.

    Drops any ``provider/`` prefix, then any Bedrock regional inference-profile
    prefix (``us.``/``eu.``/``apac.``/``global.``), then a leading
    ``anthropic.`` vendor segment, so ``bedrock/us.anthropic.claude-opus-4-8``
    and ``claude-opus-4-8`` both reduce to ``claude-opus-4-8``.
    """
    leaf = model_id.rsplit("/", 1)[-1].strip()
    leaf = _BEDROCK_REGION_PREFIX_RE.sub("", leaf)
    if leaf.lower().startswith("anthropic."):
        leaf = leaf[len("anthropic."):]
    return leaf


def _is_reasoning_model(model_id: str) -> bool:
    """True for OpenAI reasoning-class models (o-series, gpt-5.x).

    These models do not accept ``max_tokens`` or a custom ``temperature``.
    Detection is on the leaf model id with any ``provider/`` prefix stripped.
    """
    return bool(_REASONING_MODEL_RE.match(_leaf_model_id(model_id)))


def _rejects_explicit_temperature(
    model_id: str, resolved_provider: str | None
) -> bool:
    """True when passing an explicit ``temperature`` would be rejected.

    Covers OpenAI/Azure reasoning models (o-series, gpt-5.x) on any provider,
    and the newer Claude generation (Opus 4.7+, Sonnet 5+, Fable 5+) on the
    ``anthropic`` / ``bedrock`` providers (and their variants, and the
    LangChain-inferred ``provider=None`` path for a bare ``claude-…`` id).
    Everywhere else ``temperature=0.0`` is accepted and kept for determinism.
    """
    leaf = _leaf_model_id(model_id)
    if _REASONING_MODEL_RE.match(leaf):
        return True
    claude_capable_provider = resolved_provider is None or (
        "anthropic" in resolved_provider or "bedrock" in resolved_provider
    )
    if claude_capable_provider and _TEMPERATURE_DEPRECATED_CLAUDE_RE.match(leaf):
        return True
    return False


def _provider_family(resolved_provider: str | None, model_id: str) -> str:
    """Coarse provider family for request-param mapping.

    Explicit provider wins; otherwise infer from the leaf model id. Returns one
    of ``openai`` / ``anthropic`` / ``bedrock`` / ``google`` / ``ollama``.
    ``openai`` also covers Azure and OpenAI-compatible (vLLM) endpoints.
    """
    prov = (resolved_provider or "").lower()
    leaf = _leaf_model_id(model_id).lower()
    if "bedrock" in prov:
        return "bedrock"
    if "anthropic" in prov or (not prov and leaf.startswith("claude")):
        return "anthropic"
    if "google" in prov or (not prov and leaf.startswith("gemini")):
        return "google"
    if "ollama" in prov:
        return "ollama"
    return "openai"


def _is_openai_compatible_endpoint(
    resolved_provider: str | None, api_base: str | None
) -> bool:
    """True for a self-hosted OpenAI-compatible backend (e.g. vLLM, SGLang).

    The ``chat_template_kwargs`` / ``enable_thinking`` toggle is a self-hosted
    extension passed through to the server's tokenizer chat template — native
    OpenAI (``api.openai.com``) and Azure OpenAI both reject it. A custom
    ``api_base`` is the signal that we are talking to such a compatible server;
    Azure also sets a custom endpoint but is explicitly excluded because it is
    not compatible with this field.
    """
    if not api_base:
        return False
    return (resolved_provider or "").lower() != "azure_openai"


def _reasoning_control_kwargs(
    resolved_provider: str | None,
    model_id: str,
    reasoning: str,
    api_base: str | None = None,
) -> dict[str, Any]:
    """Provider-correct kwargs to disable/enable model "thinking".

    ``reasoning`` is ``auto`` (default — inject nothing), ``off`` (disable), or
    ``on`` (enable where a toggle exists). The disable kwarg differs per
    provider, so it is keyed on the provider family and an OpenAI-only key
    (``extra_body``) never leaks to another provider:

    * openai / azure_openai: native reasoning models →
      ``reasoning_effort='minimal'``; non-reasoning native OpenAI/Azure models
      have no thinking toggle → no-op.
    * vLLM / OpenAI-compatible self-hosted (custom ``api_base``) open models →
      ``extra_body={'chat_template_kwargs': {'enable_thinking': <bool>}}``.
    * anthropic → ``thinking={'type': 'disabled'}``.
    * bedrock → ``model_kwargs={'thinking': {'type': 'disabled'}}`` (InvokeModel body).
    * google_genai → ``thinking_budget=0``.
    * ollama → no toggle (no-op).
    """
    if reasoning not in ("off", "on"):
        return {}
    family = _provider_family(resolved_provider, model_id)
    # The chat-template toggle is only valid for a self-hosted OpenAI-compatible
    # server; native OpenAI/Azure reject the field (and their non-reasoning
    # models have nothing to toggle), so gate it on the endpoint.
    compat_toggle = family == "openai" and _is_openai_compatible_endpoint(
        resolved_provider, api_base
    )

    if reasoning == "off":
        if family == "openai":
            if _is_reasoning_model(model_id):
                return {"reasoning_effort": "minimal"}
            if compat_toggle:
                return {
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
                }
            return {}
        if family == "bedrock":
            # ChatBedrock drives Claude over the InvokeModel path, whose body is
            # the Anthropic Messages format — the thinking control belongs in the
            # request body via ``model_kwargs``. ``additional_model_request_fields``
            # is Converse-only and the InvokeModel API rejects it with
            # "Extra inputs are not permitted" (verified live on Opus 4.8).
            return {"model_kwargs": {"thinking": {"type": "disabled"}}}
        if family == "anthropic":
            return {"thinking": {"type": "disabled"}}
        if family == "google":
            return {"thinking_budget": 0}
        return {}

    # reasoning == "on": only the vLLM/OpenAI-compatible chat-template toggle is
    # safe to force on generically; native reasoners and other providers reason
    # by default, so leave them untouched.
    if compat_toggle and not _is_reasoning_model(model_id):
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
    return {}


try:
    from langchain.chat_models import init_chat_model
except ImportError:
    init_chat_model = None  # agentic extras not installed

_PROVIDER_IMPORT_HINTS: dict[str, tuple[str, str]] = {
    "openai": ("langchain_openai", "cisco-aibom[agentic,llm-openai]"),
    "azure_openai": ("langchain_openai", "cisco-aibom[agentic,llm-openai]"),
    "bedrock": ("langchain_aws", "cisco-aibom[agentic,llm-aws]"),
    "anthropic": ("langchain_anthropic", "cisco-aibom[agentic,llm-anthropic]"),
    "google_genai": ("langchain_google_genai", "cisco-aibom[agentic,llm-google]"),
}

# Inverted view of the hints above, keyed by the integration *module* whose
# import fails. This is how a missing provider binding is translated into the
# right ``cisco-aibom`` extra WITHOUT this code inferring the provider from the
# model name: ``init_chat_model`` performs the model->provider inference and
# then imports the integration package, so a missing binding surfaces as an
# ``ImportError`` naming the exact module (e.g. ``langchain_openai``). We map
# that module back to its install target.
_MODULE_INSTALL_TARGETS: dict[str, str] = {
    module: install_target for module, install_target in _PROVIDER_IMPORT_HINTS.values()
}


def _missing_binding_module(exc: BaseException) -> str | None:
    """Return the top-level module a failed import was looking for.

    Walks the exception's cause/context chain for the original
    ``ModuleNotFoundError`` (``init_chat_model`` re-raises ``ImportError``
    ``from`` it) and returns its top-level module name — e.g. ``langchain_openai``
    for a failed ``import langchain_openai.chat_models``. Returns ``None`` when no
    module name can be recovered.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = getattr(current, "name", None)
        if isinstance(current, ModuleNotFoundError) and name:
            return name.split(".", 1)[0]
        current = current.__cause__ or current.__context__
    return None


def _provider_runtime_hint(exc: ImportError) -> str:
    """Translate a provider-binding import failure into an actionable message.

    Maps the missing integration module to the ``cisco-aibom`` extra that ships
    it. Falls back to listing the bundled provider extras when the module is one
    this project does not package (a provider ``langchain`` supports but the AI
    BOM CLI does not bundle an extra for).
    """
    module = _missing_binding_module(exc)
    install_target = _MODULE_INSTALL_TARGETS.get(module or "")
    if install_target:
        return (
            f"The LLM provider integration for this model is not installed "
            f"(missing '{module}'). "
            f'Install it with: uv tool install "{install_target}"'
        )
    bundled = ", ".join(sorted(set(_MODULE_INSTALL_TARGETS.values())))
    missing = f" (missing '{module}')" if module else ""
    return (
        "The LLM provider integration for this model is not installed"
        f"{missing}. Install the provider extra that matches your "
        f"--llm-model — one of: {bundled}."
    )


def resolve_provider(model_string: str, provider: str | None = None) -> str | None:
    """Resolve the effective model provider from explicit or legacy syntax."""
    if provider:
        return provider
    if "/" in model_string:
        resolved_provider, _, _ = model_string.partition("/")
        return resolved_provider or None
    return None


def ensure_llm_runtime_available(
    model_string: str,
    *,
    provider: str | None = None,
) -> str | None:
    """Fail fast when the required agentic or provider extras are missing."""
    if init_chat_model is None:
        raise ImportError(
            "LLM-assisted analysis requires the agentic extras. "
            'Install with: uv tool install "cisco-aibom[agentic]"'
        )

    resolved_provider = resolve_provider(model_string, provider)

    if resolved_provider:
        # Provider is known explicitly (``--llm-provider`` or a ``provider/``
        # prefix). For a provider we bundle an extra for, do a cheap targeted
        # import check with a precise hint. Providers we do not bundle (e.g.
        # ``ollama``) are the caller's responsibility — don't second-guess them.
        provider_hint = _PROVIDER_IMPORT_HINTS.get(resolved_provider)
        if provider_hint:
            module_name, install_target = provider_hint
            try:
                importlib.import_module(module_name)
            except ImportError as exc:
                raise ImportError(
                    f"LLM provider '{resolved_provider}' requires additional "
                    f"runtime support. "
                    f'Install with: uv tool install "{install_target}"'
                ) from exc
        return resolved_provider

    # No provider we can resolve ourselves — a name-inferred model such as a
    # bare ``gpt-4o``. Rather than re-implement langchain's model->provider
    # inference here, let ``init_chat_model`` infer the provider and import its
    # integration package; that import happens before the model is constructed,
    # so no credentials are needed. Only a *missing binding* (ImportError) is
    # ours to translate — any other error (e.g. an un-inferable name) means the
    # binding is present and is surfaced later with the real configuration.
    try:
        init_chat_model(model_string, model_provider=provider)
    except ImportError as exc:
        raise ImportError(_provider_runtime_hint(exc)) from exc
    except Exception:  # noqa: BLE001 - binding imported; other failures handled later
        pass
    return None


def build_chat_model(
    model_string: str,
    *,
    provider: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    api_version: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    rate_limiter: Any | None = None,
    reasoning: str = "auto",
    init_kwargs_extra: dict[str, Any] | None = None,
) -> Any:
    """Build a LangChain ``BaseChatModel`` with unified provider routing.

    Provider resolution order:

    1. Explicit *provider* argument  (``--llm-provider`` CLI flag).
    2. ``provider/model-id`` slash convention in *model_string*
       (backward-compatible with the former LiteLLM convention).
    3. Let LangChain's ``init_chat_model`` infer from the model name
       (works for ``gpt-5.4``, ``claude-sonnet-4-20250514``, etc.).

    Parameters
    ----------
    model_string:
        Model identifier, optionally prefixed with ``provider/``
        (e.g. ``"bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"``).
    provider:
        Explicit LangChain provider name (e.g. ``"bedrock"``,
        ``"openai"``, ``"azure_openai"``, ``"ollama"``).
    api_key, api_base, api_version:
        Connection credentials forwarded from ``--llm-*`` CLI flags.
    temperature:
        Sampling temperature (default ``0.0``).
    max_tokens:
        Maximum tokens for the response.  ``None`` lets the provider
        choose its own default.
    rate_limiter:
        Optional ``langchain_core.rate_limiters.BaseRateLimiter``.
    reasoning:
        ``auto`` (default, no change), ``off``, or ``on``. ``off`` emits the
        provider-correct thinking-disable parameter (see
        :func:`_reasoning_control_kwargs`) so reasoning-class models can be
        driven without every batch timing out.
    init_kwargs_extra:
        Optional dict of provider-specific init kwargs merged verbatim into the
        ``init_chat_model`` call, LAST, so it overrides everything above. Keys
        are provider-specific (e.g. ``extra_body``, ``model_kwargs``,
        ``additional_model_request_fields``, ``reasoning_effort``).
    """
    resolved_provider = ensure_llm_runtime_available(
        model_string,
        provider=provider,
    )

    init_kwargs: dict[str, Any] = {}

    model_id = model_string
    if "/" in model_string:
        inferred_provider, _, legacy_model_id = model_string.partition("/")
        if not provider or provider == inferred_provider:
            model_id = legacy_model_id

    if resolved_provider:
        init_kwargs["model_provider"] = resolved_provider

    if api_base:
        if resolved_provider == "azure_openai":
            init_kwargs["azure_endpoint"] = api_base
        else:
            init_kwargs["base_url"] = api_base

    if api_key:
        init_kwargs["api_key"] = api_key

    if api_version and resolved_provider == "azure_openai":
        init_kwargs["api_version"] = api_version

    # ``max_tokens`` and ``temperature`` are independent axes.
    #
    # max_tokens: OpenAI reasoning-class models (o-series, gpt-5.x) reject
    # ``max_tokens`` and require ``max_completion_tokens``; every other
    # provider uses ``max_tokens``.
    if max_tokens is not None:
        if _is_reasoning_model(model_id):
            init_kwargs["max_completion_tokens"] = max_tokens
        else:
            init_kwargs["max_tokens"] = max_tokens

    # temperature: pin 0.0 for determinism (BOM reproducibility) wherever the
    # provider/model accepts it, but OMIT it where an explicit temperature is
    # rejected — OpenAI/Azure reasoning models and newer Claude (Opus 4.7+,
    # Sonnet 5+, Fable 5+) on the anthropic/bedrock providers.
    # Omission lets the provider apply its own required default.
    if not _rejects_explicit_temperature(model_id, resolved_provider):
        init_kwargs["temperature"] = temperature

    if rate_limiter is not None:
        init_kwargs["rate_limiter"] = rate_limiter

    # Provider-correct reasoning/thinking control (--llm-reasoning). ``api_base``
    # distinguishes a self-hosted OpenAI-compatible endpoint (vLLM) from native
    # OpenAI/Azure, which reject the chat-template toggle.
    init_kwargs.update(
        _reasoning_control_kwargs(
            resolved_provider, model_id, reasoning, api_base=api_base
        )
    )

    # Generic per-provider passthrough (--llm-init-kwargs): merged verbatim and
    # LAST so power users can override anything above. Keys are provider-specific.
    if init_kwargs_extra:
        init_kwargs.update(init_kwargs_extra)

    try:
        return init_chat_model(model_id, **init_kwargs)
    except ImportError as exc:
        # Definitive guard covering every resolution path: a name-inferred
        # provider whose integration package is missing reaches us here. The
        # pre-flight in ``ensure_llm_runtime_available`` catches the common case
        # earlier, but translating here too keeps ``build_chat_model``
        # self-contained. Map the missing module to the matching extra.
        raise ImportError(_provider_runtime_hint(exc)) from exc
