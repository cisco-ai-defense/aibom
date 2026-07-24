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

"""Public schema-v2 vocabulary metadata used by offline KB inspection."""

from __future__ import annotations

from typing import Final

SCHEMA_VERSION: Final[int] = 2
VOCABULARY_VERSION: Final[str] = "v2.0"

CONCEPTS: Final[tuple[str, ...]] = (
    "model",
    "embedding",
    "reranker",
    "agent",
    "tool",
    "skill",
    "prompt",
    "vector_store",
    "retriever",
    "memory",
    "guardrail",
    "evaluator",
    "mcp_server",
    "mcp_client",
    "dataset",
    "training_run",
    "model_artifact",
    "observability",
    "framework_core",
    "document_loader",
)


def schema_major(value: object) -> int | None:
    """Return the leading numeric schema component, if one is present."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip().lower()
    if text.startswith("v"):
        text = text[1:]
    first = text.split(".", 1)[0]
    return int(first) if first.isdigit() else None
