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

"""AIBOM scanner tools exposed as LangChain StructuredTool instances.

All tools wrap existing deterministic AIBOM functionality so the agentic
layer can invoke them via the Deep Agents harness.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schemas for tool arguments
# ---------------------------------------------------------------------------


class ScanDirectoryArgs(BaseModel):
    path: str = Field(description="Absolute or relative directory path to scan.")


class ResolveEnvVarArgs(BaseModel):
    var_name: str = Field(description="Environment variable name (e.g. MODEL_NAME).")
    search_paths: list[str] = Field(
        description="List of directory paths to search for definitions."
    )


class LookupModelArgs(BaseModel):
    identifier: str = Field(
        description="Model identifier (e.g. 'gpt-4o', 'meta-llama/Llama-3-70B')."
    )


class AnalyzeImportsArgs(BaseModel):
    file_path: str = Field(description="Absolute path to a Python file.")


class TraceDataFlowArgs(BaseModel):
    symbol: str = Field(description="Variable or symbol name to trace.")
    file_path: str = Field(description="Absolute path to the file containing the symbol.")


class SearchCodebaseArgs(BaseModel):
    pattern: str = Field(description="Regex or literal pattern to search for.")
    search_paths: list[str] = Field(
        description="List of directory paths to search in."
    )
    literal: bool = Field(
        default=False,
        description="If True, treat pattern as a literal string (not regex).",
    )


# ---------------------------------------------------------------------------
# Tool implementations (pure functions, no LangChain dependency)
# ---------------------------------------------------------------------------

_ENV_FILE_NAMES = frozenset({
    ".env", ".env.local", ".env.production", ".env.staging", ".env.development",
})

_ENV_CONFIG_GLOBS = (
    "docker-compose*.yml", "docker-compose*.yaml",
    "*.tfvars", "terraform.tfvars",
    "values.yaml", "values-*.yaml",
)


def scan_directory_impl(path: str) -> str:
    """Run all registered scanners on *path* and return JSON summary."""
    from ..models import ScanContext
    from ..scanners import run_scanners

    ctx = ScanContext(paths=[path])
    components, relationships = run_scanners(ctx)

    result = {
        "path": path,
        "total_components": len(components),
        "total_relationships": len(relationships),
        "components": [
            {
                "instance_id": c.instance_id,
                "name": c.name,
                "component_type": c.component_type.value,
                "file_path": c.file_path,
                "line_number": c.line_number,
                "framework": c.framework,
                "model_name": c.model_name,
                "detection_source": c.detection_source.value,
                "metadata": c.metadata,
            }
            for c in components
        ],
        "relationships": [
            {
                "source": r.source_name,
                "target": r.target_name,
                "type": r.relationship_type.value,
            }
            for r in relationships
        ],
    }
    return json.dumps(result, indent=2)


def resolve_env_var_impl(var_name: str, search_paths: list[str]) -> str:
    """Search for *var_name* definitions in env files and configs."""
    import yaml

    found: list[dict[str, Any]] = []
    pattern = re.compile(
        rf"^{re.escape(var_name)}\s*[:=]\s*(.+)$", re.MULTILINE
    )

    for base in search_paths:
        base_path = Path(base)
        if not base_path.is_dir():
            continue

        for env_name in _ENV_FILE_NAMES:
            env_file = base_path / env_name
            if env_file.is_file():
                text = env_file.read_text(errors="replace")
                for m in pattern.finditer(text):
                    val = m.group(1).strip().strip("'\"")
                    found.append({
                        "file": str(env_file),
                        "value": val,
                        "source_type": "env_file",
                    })

        for glob_pat in _ENV_CONFIG_GLOBS:
            for cfg in base_path.rglob(glob_pat):
                if not cfg.is_file():
                    continue
                text = cfg.read_text(errors="replace")
                if cfg.suffix in (".yml", ".yaml"):
                    try:
                        data = yaml.safe_load(text) or {}
                        _search_yaml_for_key(data, var_name, str(cfg), found)
                    except yaml.YAMLError:
                        pass
                for m in pattern.finditer(text):
                    val = m.group(1).strip().strip("'\"")
                    found.append({
                        "file": str(cfg),
                        "value": val,
                        "source_type": "config_file",
                    })

    if not found:
        return json.dumps({"var_name": var_name, "resolved": False, "matches": []})
    return json.dumps({"var_name": var_name, "resolved": True, "matches": found})


def _search_yaml_for_key(
    data: Any, key: str, file_path: str, found: list[dict[str, Any]], prefix: str = ""
) -> None:
    """Recursively search a parsed YAML dict for a key matching *key*."""
    if isinstance(data, dict):
        for k, v in data.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if k == key or k.upper() == key.upper():
                if isinstance(v, (str, int, float, bool)):
                    found.append({
                        "file": file_path,
                        "value": str(v),
                        "yaml_path": full_key,
                        "source_type": "yaml_value",
                    })
            _search_yaml_for_key(v, key, file_path, found, full_key)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            _search_yaml_for_key(item, key, file_path, found, f"{prefix}[{i}]")


def lookup_model_impl(identifier: str) -> str:
    """Query model registries for metadata about *identifier*."""
    from ..scanners.model_detector import _registry_lookup

    entry = _registry_lookup(identifier)
    if entry is None:
        return json.dumps({
            "identifier": identifier,
            "found": False,
            "message": f"No registry entry found for '{identifier}'.",
        })

    result: dict[str, Any] = {
        "identifier": identifier,
        "found": True,
        "provider": entry.get("provider", "unknown"),
        "family": entry.get("family", ""),
        "deprecated": entry.get("deprecated", False),
        "model_card_url": entry.get("model_card_url", ""),
        "source": entry.get("source", ""),
    }
    for extra_key in ("license", "downloads", "pipeline_tag", "hf_id"):
        if extra_key in entry:
            result[extra_key] = entry[extra_key]
    return json.dumps(result)


def analyze_imports_impl(file_path: str) -> str:
    """Run LibCST deep import analysis on a single Python file."""
    from ..cst_parser import parse_source_code

    p = Path(file_path)
    if not p.is_file() or p.suffix != ".py":
        return json.dumps({"error": f"Not a Python file: {file_path}"})

    source = p.read_text(errors="replace")
    result = parse_source_code(file_path, source)
    return json.dumps({
        "file": file_path,
        "imports": result.imports,
        "calls": [
            {
                "name": c.qualified_name,
                "line": c.line_number,
            }
            for c in result.calls[:50]
        ],
        "assignments": [
            {
                "target": a.target_qualified_name,
                "value": a.call.qualified_name,
                "line": a.line_number,
            }
            for a in result.assignments[:50]
        ],
        "decorators": [
            {
                "name": d.decorator_qualified_name,
                "line": d.line_number,
            }
            for d in result.decorators[:20]
        ],
    })


def trace_data_flow_impl(symbol: str, file_path: str) -> str:
    """Trace a variable through assignments to resolve its concrete value."""
    p = Path(file_path)
    if not p.is_file():
        return json.dumps({"error": f"File not found: {file_path}"})

    source = p.read_text(errors="replace")
    lines = source.splitlines()

    assign_re = re.compile(
        rf"^\s*{re.escape(symbol)}\s*=\s*(.+)$", re.MULTILINE
    )

    chain: list[dict[str, Any]] = []
    for m in assign_re.finditer(source):
        line_no = source[:m.start()].count("\n") + 1
        rhs = m.group(1).strip()
        chain.append({"line": line_no, "value": rhs})

    param_re = re.compile(
        rf'{re.escape(symbol)}\s*=\s*["\']([^"\']+)["\']'
    )
    literals: list[dict[str, Any]] = []
    for m in param_re.finditer(source):
        line_no = source[:m.start()].count("\n") + 1
        literals.append({"line": line_no, "value": m.group(1)})

    os_env_re = re.compile(
        rf'{re.escape(symbol)}\s*=\s*os\.(?:environ(?:\.get)?\s*[\[(]\s*["\']([^"\']+)["\']|getenv\s*\(\s*["\']([^"\']+)["\'])'
    )
    env_refs: list[dict[str, Any]] = []
    for m in os_env_re.finditer(source):
        line_no = source[:m.start()].count("\n") + 1
        env_var = m.group(1) or m.group(2)
        env_refs.append({"line": line_no, "env_var": env_var})

    return json.dumps({
        "symbol": symbol,
        "file": file_path,
        "assignments": chain,
        "string_literals": literals,
        "env_var_references": env_refs,
        "resolved": bool(literals),
        "concrete_value": literals[0]["value"] if literals else None,
    })


def search_codebase_impl(
    pattern: str, search_paths: list[str], literal: bool = False
) -> str:
    """Search across directories for a regex or literal pattern."""
    if literal:
        regex = re.compile(re.escape(pattern), re.IGNORECASE)
    else:
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            return json.dumps({"error": f"Invalid regex: {exc}"})

    matches: list[dict[str, Any]] = []
    max_matches = 100

    for base in search_paths:
        base_path = Path(base)
        if not base_path.is_dir():
            continue
        for fp in base_path.rglob("*"):
            if len(matches) >= max_matches:
                break
            if not fp.is_file() or fp.stat().st_size > 1_000_000:
                continue
            if fp.suffix in (".pyc", ".pyo", ".so", ".dll", ".exe", ".bin"):
                continue
            try:
                text = fp.read_text(errors="replace")
            except (OSError, UnicodeDecodeError):
                continue
            for m in regex.finditer(text):
                line_no = text[:m.start()].count("\n") + 1
                line_text = text.splitlines()[line_no - 1] if line_no <= len(text.splitlines()) else ""
                matches.append({
                    "file": str(fp),
                    "line": line_no,
                    "match": m.group()[:200],
                    "context": line_text[:200],
                })
                if len(matches) >= max_matches:
                    break

    return json.dumps({
        "pattern": pattern,
        "total_matches": len(matches),
        "truncated": len(matches) >= max_matches,
        "matches": matches,
    })


# ---------------------------------------------------------------------------
# LangChain StructuredTool factory (lazy — only called when deepagents used)
# ---------------------------------------------------------------------------


def build_tools() -> list[Any]:
    """Create LangChain StructuredTool instances wrapping the AIBOM tools.

    This function is called lazily only when ``--agent-model`` is specified,
    so the ``langchain_core`` import happens only when needed.
    """
    from langchain_core.tools import StructuredTool

    return [
        StructuredTool.from_function(
            name="scan_directory",
            description=(
                "Run all AIBOM deterministic scanners on a directory. "
                "Returns detected AI components and relationships as JSON."
            ),
            func=scan_directory_impl,
            args_schema=ScanDirectoryArgs,
        ),
        StructuredTool.from_function(
            name="resolve_env_var",
            description=(
                "Search for an environment variable definition across multiple "
                "directory paths. Checks .env files, docker-compose, Terraform "
                "tfvars, Helm values.yaml, and other config files."
            ),
            func=resolve_env_var_impl,
            args_schema=ResolveEnvVarArgs,
        ),
        StructuredTool.from_function(
            name="lookup_model",
            description=(
                "Query model registries (LiteLLM catalog, HuggingFace Hub, "
                "built-in) for metadata about a model identifier. Returns "
                "provider, license, deprecation status, and model card URL."
            ),
            func=lookup_model_impl,
            args_schema=LookupModelArgs,
        ),
        StructuredTool.from_function(
            name="analyze_imports",
            description=(
                "Run deep import analysis on a Python file using LibCST. "
                "Returns imports, calls, assignments, and decorators — useful "
                "for disambiguating which framework a symbol comes from."
            ),
            func=analyze_imports_impl,
            args_schema=AnalyzeImportsArgs,
        ),
        StructuredTool.from_function(
            name="trace_data_flow",
            description=(
                "Trace a variable/symbol through assignments in a file to "
                "resolve its concrete value. Useful when a model name is "
                "passed through multiple variables or loaded from env vars."
            ),
            func=trace_data_flow_impl,
            args_schema=TraceDataFlowArgs,
        ),
        StructuredTool.from_function(
            name="search_codebase",
            description=(
                "Search across all input directories for a regex or literal "
                "pattern. Returns matching file paths, line numbers, and "
                "context snippets."
            ),
            func=search_codebase_impl,
            args_schema=SearchCodebaseArgs,
        ),
    ]
