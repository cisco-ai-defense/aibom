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

"""Workflow Scanner -- detects AI-related nodes in low-code/no-code automation
workflow definitions.

Supports:
* n8n workflow JSON (``*.n8n.json``, exported workflows)
* Dify workflow YAML
* Flowise chatflow JSON
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..models import AIComponent, AIComponentType, ComponentRelationship
from ..models.enums import DetectionSource
from ..models.scan import ScanContext
from .base import BaseScanner
from .file_cache import read_text_cached

_LOGGER = logging.getLogger(__name__)

_N8N_AI_NODE_TYPES: frozenset[str] = frozenset({
    "@n8n/n8n-nodes-langchain.chainLlm",
    "@n8n/n8n-nodes-langchain.lmChatOpenAi",
    "@n8n/n8n-nodes-langchain.lmChatAnthropic",
    "@n8n/n8n-nodes-langchain.lmChatOllama",
    "@n8n/n8n-nodes-langchain.lmChatGoogleGemini",
    "@n8n/n8n-nodes-langchain.lmChatAzureOpenAi",
    "@n8n/n8n-nodes-langchain.lmChatMistralCloud",
    "@n8n/n8n-nodes-langchain.agent",
    "@n8n/n8n-nodes-langchain.toolWorkflow",
    "@n8n/n8n-nodes-langchain.toolCode",
    "@n8n/n8n-nodes-langchain.memoryBufferWindow",
    "@n8n/n8n-nodes-langchain.memoryPostgresChat",
    "@n8n/n8n-nodes-langchain.embeddingsOpenAi",
    "@n8n/n8n-nodes-langchain.vectorStoreSupabase",
    "@n8n/n8n-nodes-langchain.vectorStorePinecone",
    "@n8n/n8n-nodes-langchain.vectorStoreQdrant",
    "@n8n/n8n-nodes-langchain.retrieverVectorStore",
    "@n8n/n8n-nodes-langchain.textSplitterRecursiveCharacterTextSplitter",
    "n8n-nodes-base.openAi",
    "n8n-nodes-base.httpRequest",
})

_N8N_NODE_TYPE_MAP: dict[str, AIComponentType] = {
    "lmChat": AIComponentType.MODEL,
    "agent": AIComponentType.AGENT,
    "tool": AIComponentType.TOOL,
    "memory": AIComponentType.MEMORY,
    "embedding": AIComponentType.EMBEDDING,
    "vectorStore": AIComponentType.VECTOR_STORE,
    "retriever": AIComponentType.RETRIEVER,
    "chain": AIComponentType.AGENT,
}

_AI_KEYWORDS_IN_NODE = frozenset({
    "openai", "anthropic", "ollama", "gemini", "azure", "mistral",
    "langchain", "llm", "gpt", "claude", "embedding", "vector",
})


class WorkflowScanner(BaseScanner):
    name = "workflow_scanner"

    def supports(self, context: ScanContext) -> bool:
        return True

    def scan(
        self, context: ScanContext,
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        components: list[AIComponent] = []
        idx = context.file_index()

        json_files: list[Path] = []
        if idx:
            for entry in idx.get(".json", []):
                json_files.append(entry.path)
        else:
            for scan_path in context.paths:
                root = Path(scan_path)
                if root.is_file() and root.suffix == ".json":
                    json_files.append(root)
                elif root.is_dir():
                    json_files.extend(root.rglob("*.json"))

        for jf in json_files:
            try:
                text = read_text_cached(jf)
            except Exception:
                continue
            try:
                data = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(data, dict):
                continue

            if _is_n8n_workflow(data):
                components.extend(_scan_n8n(jf, data))

        return components, []


def _is_n8n_workflow(data: dict[str, Any]) -> bool:
    return "nodes" in data and isinstance(data.get("nodes"), list)


def _scan_n8n(wf: Path, data: dict[str, Any]) -> list[AIComponent]:
    components: list[AIComponent] = []
    nodes = data.get("nodes", [])

    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = node.get("type", "")
        node_name = node.get("name", node_type)

        is_ai_node = node_type in _N8N_AI_NODE_TYPES
        if not is_ai_node:
            type_lower = node_type.lower()
            is_ai_node = any(kw in type_lower for kw in _AI_KEYWORDS_IN_NODE)

        if not is_ai_node:
            continue

        comp_type = _classify_n8n_node(node_type)
        params = node.get("parameters", {})
        model_name = _extract_n8n_model(params)
        meta: dict[str, Any] = {
            "workflow_type": "n8n",
            "n8n_node_type": node_type,
        }
        if model_name:
            meta["model_name"] = model_name

        components.append(
            AIComponent(
                name=node_name,
                component_type=comp_type,
                file_path=str(wf),
                line_number=0,
                model_name=model_name,
                framework="n8n",
                detection_source=DetectionSource.CONFIG_FILE,
                metadata=meta,
            )
        )

    return components


def _classify_n8n_node(node_type: str) -> AIComponentType:
    type_lower = node_type.lower()
    for keyword, comp_type in _N8N_NODE_TYPE_MAP.items():
        if keyword.lower() in type_lower:
            return comp_type
    if "openai" in type_lower or "anthropic" in type_lower:
        return AIComponentType.MODEL
    return AIComponentType.OTHER


def _extract_n8n_model(params: dict[str, Any]) -> str | None:
    for key in ("model", "modelId", "model_name"):
        val = params.get(key)
        if isinstance(val, str) and val:
            return val
    options = params.get("options", {})
    if isinstance(options, dict):
        for key in ("model", "modelId"):
            val = options.get(key)
            if isinstance(val, str) and val:
                return val
    return None
