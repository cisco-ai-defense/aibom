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

"""Parse framework-specific configuration files (langgraph.json, etc.)
to extract AI component metadata not visible in Python source alone."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .structures import (
    AssignmentObservation,
    CallObservation,
    CodeAnalysisResult,
)

LOGGER = logging.getLogger(__name__)


def parse_langgraph_json(project_root: Path) -> Optional[CodeAnalysisResult]:
    """Parse ``langgraph.json`` at *project_root* and emit synthetic observations.

    Extracts:
    - ``graphs`` entries as agent entrypoints
    - ``store.index.embed`` as an embedding model component
    - ``store.index.dims`` as metadata on the embedding
    """
    config_path = project_root / "langgraph.json"
    if not config_path.is_file():
        return None

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("Failed to parse %s: %s", config_path, exc)
        return None

    result = CodeAnalysisResult(file_path=str(config_path))
    line_counter = 1

    graphs: Dict[str, str] = data.get("graphs", {})
    for graph_name, entrypoint in graphs.items():
        result.assignments.append(
            AssignmentObservation(
                target_qualified_name=graph_name,
                call=CallObservation(
                    qualified_name="langgraph.graph.StateGraph.compile",
                    arguments={"entrypoint": entrypoint},
                    line_number=line_counter,
                    raw_code=f'# langgraph.json graphs["{graph_name}"] = "{entrypoint}"',
                ),
                line_number=line_counter,
            )
        )
        line_counter += 1

    store_config: Dict[str, Any] = data.get("store", {})
    index_config: Dict[str, Any] = store_config.get("index", {})
    embed_value = index_config.get("embed")
    dims_value = index_config.get("dims")

    if embed_value:
        embed_args: Dict[str, Any] = {"model": embed_value}
        if dims_value is not None:
            embed_args["dims"] = dims_value

        result.assignments.append(
            AssignmentObservation(
                target_qualified_name="langgraph_config_embedding",
                call=CallObservation(
                    qualified_name="langchain_openai.OpenAIEmbeddings",
                    arguments=embed_args,
                    line_number=line_counter,
                    raw_code=f"# langgraph.json store.index.embed = {embed_value}",
                ),
                line_number=line_counter,
            )
        )
        line_counter += 1

    if result.assignments or result.calls:
        LOGGER.info(
            "Extracted %d components from langgraph.json",
            len(result.assignments) + len(result.calls),
        )
        return result

    return None


_AI_PACKAGES: Dict[str, str] = {
    "@langchain/openai": "langchainjs",
    "@langchain/core": "langchainjs",
    "@langchain/langgraph": "langchainjs",
    "@langchain/anthropic": "langchainjs",
    "@langchain/community": "langchainjs",
    "langchain": "langchainjs",
    "ai": "vercel_ai",
    "@ai-sdk/openai": "vercel_ai",
    "@ai-sdk/anthropic": "vercel_ai",
    "@ai-sdk/google": "vercel_ai",
    "openai": "openai_js",
    "@anthropic-ai/sdk": "anthropic_js",
    "@tensorflow/tfjs": "tensorflowjs",
    "@tensorflow/tfjs-node": "tensorflowjs",
    "@huggingface/transformers": "transformersjs",
}


def parse_package_json(project_root: Path) -> Optional[CodeAnalysisResult]:
    """Parse ``package.json`` to detect AI SDK dependencies."""
    pkg_path = project_root / "package.json"
    if not pkg_path.is_file():
        return None

    try:
        data = json.loads(pkg_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("Failed to parse %s: %s", pkg_path, exc)
        return None

    all_deps: Dict[str, Any] = {}
    for key in ("dependencies", "devDependencies"):
        all_deps.update(data.get(key, {}))

    result = CodeAnalysisResult(file_path=str(pkg_path))
    line_counter = 1

    for pkg_name, framework in _AI_PACKAGES.items():
        if pkg_name in all_deps:
            version = all_deps[pkg_name]
            result.assignments.append(
                AssignmentObservation(
                    target_qualified_name=f"pkg_{pkg_name}",
                    call=CallObservation(
                        qualified_name=f"{pkg_name}.default",
                        arguments={"version": version},
                        line_number=line_counter,
                        raw_code=f'# package.json dependency: "{pkg_name}": "{version}"',
                    ),
                    line_number=line_counter,
                )
            )
            line_counter += 1

    if result.assignments:
        LOGGER.info(
            "Detected %d AI SDK dependency(ies) from package.json",
            len(result.assignments),
        )
        return result

    return None


def parse_project_configs(project_root: Path) -> List[CodeAnalysisResult]:
    """Parse all supported config files and return any synthetic analysis results."""
    results: List[CodeAnalysisResult] = []

    lg_result = parse_langgraph_json(project_root)
    if lg_result:
        results.append(lg_result)

    pkg_result = parse_package_json(project_root)
    if pkg_result:
        results.append(pkg_result)

    return results
