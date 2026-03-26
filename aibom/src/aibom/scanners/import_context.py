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

"""Shared import-context checker for determining whether a file has AI/ML framework imports.

Scanners use this to gate ambiguous pattern matches: if the file has no AI
framework imports, a `.fit()` call is far more likely to be sklearn preprocessing
or a pandas operation than an ML training run.
"""

from __future__ import annotations

import re

_ML_IMPORT_RE = re.compile(
    r"(?:^|\n)\s*(?:from|import)\s+"
    r"(?:torch|tensorflow|tf|keras|transformers|accelerate|"
    r"huggingface_hub|datasets|tokenizers|peft|trl|"
    r"lightning|pytorch_lightning|"
    r"jax|flax|optax|"
    r"sklearn|scikit.learn|xgboost|lightgbm|catboost)\b",
    re.MULTILINE,
)

_LLM_IMPORT_RE = re.compile(
    r"(?:^|\n)\s*(?:from|import)\s+"
    r"(?:openai|anthropic|cohere|google\.generativeai|"
    r"langchain|langchain_core|langchain_openai|langchain_anthropic|langchain_community|"
    r"llama_index|llamaindex|"
    r"litellm|"
    r"crewai|autogen|dspy|"
    r"vertexai|google\.cloud\.aiplatform)\b",
    re.MULTILINE,
)

_AGENT_IMPORT_RE = re.compile(
    r"(?:^|\n)\s*(?:from|import)\s+"
    r"(?:langchain|langchain_core|langchain_openai|"
    r"crewai|autogen|smolagents|"
    r"llama_index|llamaindex|"
    r"langgraph|deepagents|"
    r"swarm|agency_swarm|"
    r"agno|phidata|pydantic_ai)\b",
    re.MULTILINE,
)

_DATA_IMPORT_RE = re.compile(
    r"(?:^|\n)\s*(?:from|import)\s+"
    r"(?:datasets|tensorflow_datasets|torchvision|torchtext|"
    r"dvc|great_expectations|"
    r"apache_beam|pyspark\.ml)\b",
    re.MULTILINE,
)

_EXPERIMENT_IMPORT_RE = re.compile(
    r"(?:^|\n)\s*(?:from|import)\s+"
    r"(?:wandb|mlflow|neptune|comet_ml|clearml|"
    r"tensorboard|torch\.utils\.tensorboard|"
    r"trackio|swanlab|aim)\b",
    re.MULTILINE,
)

_MCP_IMPORT_RE = re.compile(
    r"(?:^|\n)\s*(?:from|import)\s+"
    r"(?:mcp|mcp\.server|mcp\.client|"
    r"langchain_mcp_adapters|fastmcp)\b",
    re.MULTILINE,
)


def has_ml_imports(text: str) -> bool:
    return bool(_ML_IMPORT_RE.search(text))


def has_llm_imports(text: str) -> bool:
    return bool(_LLM_IMPORT_RE.search(text))


def has_agent_imports(text: str) -> bool:
    return bool(_AGENT_IMPORT_RE.search(text))


def has_data_imports(text: str) -> bool:
    return bool(_DATA_IMPORT_RE.search(text))


def has_experiment_imports(text: str) -> bool:
    return bool(_EXPERIMENT_IMPORT_RE.search(text))


def has_mcp_imports(text: str) -> bool:
    return bool(_MCP_IMPORT_RE.search(text))


_CACHE_IMPORT_RE = re.compile(
    r"(?:^|\n)\s*(?:from|import)\s+"
    r"(?:redis|memcache|pymemcache|cachetools|diskcache|aiocache)\b",
    re.MULTILINE,
)


def has_cache_imports(text: str) -> bool:
    return bool(_CACHE_IMPORT_RE.search(text))


def has_any_ai_imports(text: str) -> bool:
    return (
        has_ml_imports(text)
        or has_llm_imports(text)
        or has_agent_imports(text)
        or has_data_imports(text)
        or has_experiment_imports(text)
        or has_mcp_imports(text)
    )
