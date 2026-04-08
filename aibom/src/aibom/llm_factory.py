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

import logging
from typing import Any

_logger = logging.getLogger(__name__)

try:
    from langchain.chat_models import init_chat_model
except ImportError:
    init_chat_model = None  # agentic extras not installed


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
    """
    if init_chat_model is None:
        raise ImportError(
            "langchain is required for LLM features. "
            "Install with: uv pip install 'cisco-aibom[agentic]'"
        )

    init_kwargs: dict[str, Any] = {}

    model_id = model_string
    resolved_provider = provider

    if not resolved_provider and "/" in model_string:
        resolved_provider, _, model_id = model_string.partition("/")

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

    init_kwargs["temperature"] = temperature
    if max_tokens is not None:
        init_kwargs["max_tokens"] = max_tokens

    if rate_limiter is not None:
        init_kwargs["rate_limiter"] = rate_limiter

    return init_chat_model(model_id, **init_kwargs)
