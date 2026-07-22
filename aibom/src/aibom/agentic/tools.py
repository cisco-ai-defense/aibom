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
import time
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from ..cache_paths import cache_read_dirs, ensure_cache_dir

_LOGGER = logging.getLogger(__name__)

import contextvars

_batch_tool_stats: contextvars.ContextVar[dict[str, dict[str, Any]]] = (
    contextvars.ContextVar("_batch_tool_stats")
)

_allowed_search_roots: contextvars.ContextVar[tuple[str, ...] | None] = (
    contextvars.ContextVar("_allowed_search_roots", default=None)
)
_strict_tool_root_enforcement: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_strict_tool_root_enforcement", default=False
)


def set_allowed_search_roots(
    paths: list[str] | None,
) -> contextvars.Token[tuple[str, ...] | None]:
    """Restrict tools to *paths* in the current invocation context."""
    roots = (
        None if paths is None else tuple(str(Path(path).resolve()) for path in paths)
    )
    return _allowed_search_roots.set(roots)


def reset_allowed_search_roots(
    token: contextvars.Token[tuple[str, ...] | None],
) -> None:
    """Restore the approved roots that preceded an invocation."""
    _allowed_search_roots.reset(token)


def set_strict_tool_root_enforcement(value: bool) -> contextvars.Token[bool]:
    """Enable all-tool guards for an approved raw-trajectory invocation."""
    return _strict_tool_root_enforcement.set(bool(value))


def reset_strict_tool_root_enforcement(token: contextvars.Token[bool]) -> None:
    """Restore the preceding raw-trajectory guard mode."""
    _strict_tool_root_enforcement.reset(token)


def _path_is_allowed(requested: str) -> bool:
    """Return whether *requested* is inside a configured source root."""
    allowed_roots = _allowed_search_roots.get()
    if allowed_roots is None:
        return True
    if not allowed_roots:
        return False
    try:
        resolved = Path(requested).resolve()
    except (OSError, RuntimeError):
        return False
    for root in allowed_roots:
        try:
            resolved.relative_to(Path(root))
            return True
        except ValueError:
            continue
    return False


def _strict_path_is_denied(requested: str) -> bool:
    return _strict_tool_root_enforcement.get() and not _path_is_allowed(requested)


def _clamp_paths(requested: list[str], *, tool_name: str) -> list[str]:
    """Return only those paths that fall under an allowed search root.

    If no roots are configured, returns *requested* unchanged (backward compat).
    """
    allowed_roots = _allowed_search_roots.get()
    if not allowed_roots:
        return requested
    clamped = [path for path in requested if _path_is_allowed(path)]
    denied = len(requested) - len(clamped)
    if denied:
        _record_guard_denial(tool_name, denied)
    return clamped or list(allowed_roots)


def _reset_tool_stats() -> None:
    _batch_tool_stats.set({})


def get_tool_stats() -> dict[str, dict[str, Any]]:
    try:
        return dict(_batch_tool_stats.get())
    except LookupError:
        return {}


def _get_stats_dict() -> dict[str, dict[str, Any]]:
    try:
        return _batch_tool_stats.get()
    except LookupError:
        d: dict[str, dict[str, Any]] = {}
        _batch_tool_stats.set(d)
        return d


def _stats_entry(name: str) -> dict[str, Any]:
    return _get_stats_dict().setdefault(
        name,
        {"calls": 0, "total_s": 0.0, "errors": 0, "guard_denials": 0},
    )


def _record_guard_denial(tool_name: str, count: int = 1) -> None:
    _stats_entry(tool_name)["guard_denials"] += max(0, count)


def _track_tool(name: str):
    """Decorator that logs and tracks timing for each tool invocation."""

    def decorator(fn):
        def wrapper(*args, **kwargs):
            # Tool arguments can contain repository content and identifiers.
            _LOGGER.info("Tool call: %s", name)
            t0 = time.monotonic()
            try:
                result = fn(*args, **kwargs)
                elapsed = time.monotonic() - t0
                _LOGGER.info("Tool done: %s — %.2fs", name, elapsed)
                entry = _stats_entry(name)
                entry["calls"] += 1
                entry["total_s"] += elapsed
                return result
            except Exception as exc:
                elapsed = time.monotonic() - t0
                _LOGGER.warning("Tool error: %s — %.2fs — %s", name, elapsed, exc)
                entry = _stats_entry(name)
                entry["calls"] += 1
                entry["total_s"] += elapsed
                entry["errors"] += 1
                raise

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper

    return decorator


def _summarize_args(args: tuple, kwargs: dict) -> str:
    parts = []
    for a in args:
        s = str(a)
        parts.append(s[:80] + "…" if len(s) > 80 else s)
    for k, v in kwargs.items():
        s = str(v)
        parts.append(f"{k}={s[:60]}…" if len(s) > 60 else f"{k}={s}")
    return ", ".join(parts) or "(none)"


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
        description="Model identifier (e.g. 'gpt-5.4', 'meta-llama/Llama-3-70B')."
    )


class AnalyzeImportsArgs(BaseModel):
    file_path: str = Field(description="Absolute path to a Python file.")


class TraceDataFlowArgs(BaseModel):
    symbol: str = Field(description="Variable or symbol name to trace.")
    file_path: str = Field(
        description="Absolute path to the file containing the symbol."
    )


class SearchCodebaseArgs(BaseModel):
    pattern: str = Field(description="Regex or literal pattern to search for.")
    search_paths: list[str] = Field(description="List of directory paths to search in.")
    literal: bool = Field(
        default=False,
        description="If True, treat pattern as a literal string (not regex).",
    )


# ---------------------------------------------------------------------------
# Tool implementations (pure functions, no LangChain dependency)
# ---------------------------------------------------------------------------

_ENV_FILE_NAMES = frozenset(
    {
        ".env",
        ".env.local",
        ".env.production",
        ".env.staging",
        ".env.development",
    }
)

_ENV_CONFIG_GLOBS = (
    "docker-compose*.yml",
    "docker-compose*.yaml",
    "*.tfvars",
    "terraform.tfvars",
    "values.yaml",
    "values-*.yaml",
)


@_track_tool("scan_directory")
def scan_directory_impl(path: str) -> str:
    """Run all registered scanners on *path* and return JSON summary."""
    if _strict_path_is_denied(path):
        _record_guard_denial("scan_directory")
        return json.dumps({"error": "path is outside the approved source root"})
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


@_track_tool("resolve_env_var")
def resolve_env_var_impl(var_name: str, search_paths: list[str]) -> str:
    """Search for *var_name* definitions in env files and configs."""
    search_paths = _clamp_paths(search_paths, tool_name="resolve_env_var")
    import yaml

    found: list[dict[str, Any]] = []
    pattern = re.compile(rf"^{re.escape(var_name)}\s*[:=]\s*(.+)$", re.MULTILINE)

    for base in search_paths:
        base_path = Path(base)
        if not base_path.is_dir():
            continue

        for env_name in _ENV_FILE_NAMES:
            env_file = base_path / env_name
            if _strict_path_is_denied(str(env_file)):
                _record_guard_denial("resolve_env_var")
                continue
            if env_file.is_file():
                text = env_file.read_text(errors="replace")
                for m in pattern.finditer(text):
                    val = m.group(1).strip().strip("'\"")
                    found.append(
                        {
                            "file": str(env_file),
                            "value": val,
                            "source_type": "env_file",
                        }
                    )

        for glob_pat in _ENV_CONFIG_GLOBS:
            for cfg in base_path.rglob(glob_pat):
                if _strict_path_is_denied(str(cfg)):
                    _record_guard_denial("resolve_env_var")
                    continue
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
                    found.append(
                        {
                            "file": str(cfg),
                            "value": val,
                            "source_type": "config_file",
                        }
                    )

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
                    found.append(
                        {
                            "file": file_path,
                            "value": str(v),
                            "yaml_path": full_key,
                            "source_type": "yaml_value",
                        }
                    )
            _search_yaml_for_key(v, key, file_path, found, full_key)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            _search_yaml_for_key(item, key, file_path, found, f"{prefix}[{i}]")


@_track_tool("lookup_model")
def lookup_model_impl(identifier: str) -> str:
    """Query model registries for metadata about *identifier*."""
    from ..scanners.model_detector import registry_lookup

    entry = registry_lookup(identifier)
    if entry is None:
        return json.dumps(
            {
                "identifier": identifier,
                "found": False,
                "message": f"No registry entry found for '{identifier}'.",
            }
        )

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


@_track_tool("analyze_imports")
def analyze_imports_impl(file_path: str) -> str:
    """Run LibCST deep import analysis on a single Python file."""
    if _strict_path_is_denied(file_path):
        _record_guard_denial("analyze_imports")
        return json.dumps({"error": "path is outside the approved source root"})
    from ..cst_parser import parse_source_code

    p = Path(file_path)
    if not p.is_file() or p.suffix != ".py":
        return json.dumps({"error": f"Not a Python file: {file_path}"})

    source = p.read_text(errors="replace")
    result = parse_source_code(file_path, source)
    return json.dumps(
        {
            "file": file_path,
            "imports": [
                entry[1] if isinstance(entry, tuple) else entry
                for entry in result.imports
            ],
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
        }
    )


@_track_tool("trace_data_flow")
def trace_data_flow_impl(symbol: str, file_path: str) -> str:
    """Trace a variable through assignments to resolve its concrete value."""
    if _strict_path_is_denied(file_path):
        _record_guard_denial("trace_data_flow")
        return json.dumps({"error": "path is outside the approved source root"})
    p = Path(file_path)
    if not p.is_file():
        return json.dumps({"error": f"File not found: {file_path}"})

    source = p.read_text(errors="replace")
    lines = source.splitlines()

    assign_re = re.compile(rf"^\s*{re.escape(symbol)}\s*=\s*(.+)$", re.MULTILINE)

    chain: list[dict[str, Any]] = []
    for m in assign_re.finditer(source):
        line_no = source[: m.start()].count("\n") + 1
        rhs = m.group(1).strip()
        chain.append({"line": line_no, "value": rhs})

    param_re = re.compile(rf'{re.escape(symbol)}\s*=\s*["\']([^"\']+)["\']')
    literals: list[dict[str, Any]] = []
    for m in param_re.finditer(source):
        line_no = source[: m.start()].count("\n") + 1
        literals.append({"line": line_no, "value": m.group(1)})

    os_env_re = re.compile(
        rf'{re.escape(symbol)}\s*=\s*os\.(?:environ(?:\.get)?\s*[\[(]\s*["\']([^"\']+)["\']|getenv\s*\(\s*["\']([^"\']+)["\'])'
    )
    env_refs: list[dict[str, Any]] = []
    for m in os_env_re.finditer(source):
        line_no = source[: m.start()].count("\n") + 1
        env_var = m.group(1) or m.group(2)
        env_refs.append({"line": line_no, "env_var": env_var})

    return json.dumps(
        {
            "symbol": symbol,
            "file": file_path,
            "assignments": chain,
            "string_literals": literals,
            "env_var_references": env_refs,
            "resolved": bool(literals),
            "concrete_value": literals[0]["value"] if literals else None,
        }
    )


@_track_tool("search_codebase")
def search_codebase_impl(
    pattern: str, search_paths: list[str], literal: bool = False
) -> str:
    """Search across directories for a regex or literal pattern."""
    search_paths = _clamp_paths(search_paths, tool_name="search_codebase")
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
            if _strict_path_is_denied(str(fp)):
                _record_guard_denial("search_codebase")
                continue
            if not fp.is_file() or fp.stat().st_size > 1_000_000:
                continue
            if fp.suffix in (".pyc", ".pyo", ".so", ".dll", ".exe", ".bin"):
                continue
            try:
                text = fp.read_text(errors="replace")
            except (OSError, UnicodeDecodeError):
                continue
            file_lines = text.split("\n")
            for m in regex.finditer(text):
                line_no = text[: m.start()].count("\n") + 1
                line_text = (
                    file_lines[line_no - 1] if line_no <= len(file_lines) else ""
                )
                matches.append(
                    {
                        "file": str(fp),
                        "line": line_no,
                        "match": m.group()[:200],
                        "context": line_text[:200],
                    }
                )
                if len(matches) >= max_matches:
                    break

    return json.dumps(
        {
            "pattern": pattern,
            "total_matches": len(matches),
            "truncated": len(matches) >= max_matches,
            "matches": matches,
        }
    )


# ---------------------------------------------------------------------------
# search_package_info — query package registries for AI relevance
# ---------------------------------------------------------------------------


class SearchPackageInfoArgs(BaseModel):
    package_name: str = Field(
        description="Package name (e.g. 'openai', 'langchain', 'chromadb')"
    )
    ecosystem: str = Field(
        default="pypi",
        description="Package ecosystem: 'pypi', 'npm', or 'go'",
    )


def _read_pkg_cache(name: str, ecosystem: str) -> dict[str, Any] | None:
    for cache_root in cache_read_dirs("packages"):
        cache_file = cache_root / ecosystem / f"{name}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                age_days = (time.time() - cache_file.stat().st_mtime) / 86400
                if age_days < 7:
                    data["is_cached"] = True
                    return data
            except (json.JSONDecodeError, OSError):
                pass
    return None


def _write_pkg_cache(name: str, ecosystem: str, data: dict[str, Any]) -> None:
    cache_file = ensure_cache_dir("packages") / ecosystem / f"{name}.json"
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data))
    except OSError:
        pass


def _fetch_pypi(name: str) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    url = f"https://pypi.org/pypi/{name}/json"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        info = data.get("info", {})
        return {
            "name": info.get("name", name),
            "summary": info.get("summary", ""),
            "description": (info.get("description") or "")[:500],
            "keywords": info.get("keywords") or "",
            "classifiers": info.get("classifiers", []),
            "home_page": info.get("home_page") or info.get("project_url", ""),
        }
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        return _fetch_pip_fallback(name, str(exc))


def _fetch_npm(name: str) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    url = f"https://registry.npmjs.org/{name}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        return {
            "name": data.get("name", name),
            "summary": data.get("description", ""),
            "description": data.get("description", ""),
            "keywords": data.get("keywords", []),
            "classifiers": [],
            "home_page": data.get("homepage", ""),
        }
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return {"name": name, "error": "npm registry unreachable"}


def _fetch_go(module: str) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    url = f"https://proxy.golang.org/{module}/@latest"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        return {
            "name": module,
            "summary": f"Go module {module}",
            "description": f"Version: {data.get('Version', 'unknown')}",
            "keywords": "",
            "classifiers": [],
            "home_page": f"https://pkg.go.dev/{module}",
        }
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return {"name": module, "error": "Go proxy unreachable"}


def _fetch_pip_fallback(name: str, original_error: str) -> dict[str, Any]:
    """Fall back to locally installed package metadata via pip show."""
    import subprocess

    try:
        result = subprocess.run(
            ["pip", "show", "--verbose", name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.splitlines()
            info: dict[str, str] = {}
            for line in lines:
                if ": " in line:
                    key, _, val = line.partition(": ")
                    info[key.strip()] = val.strip()
            return {
                "name": info.get("Name", name),
                "summary": info.get("Summary", ""),
                "description": info.get("Summary", ""),
                "keywords": "",
                "classifiers": [
                    c.strip()
                    for c in lines
                    if c.strip().startswith(("Topic", "Intended", "License"))
                ],
                "home_page": info.get("Home-page", ""),
            }
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return {"name": name, "error": f"Could not fetch package info: {original_error}"}


def search_package_info_impl(package_name: str, ecosystem: str = "pypi") -> str:
    """Query a package registry and return metadata for AI-relevance assessment."""
    t0 = time.monotonic()

    cached = _read_pkg_cache(package_name, ecosystem)
    if cached:
        elapsed = time.monotonic() - t0
        entry = _stats_entry("search_package_info")
        entry["calls"] += 1
        entry["total_s"] += elapsed
        return json.dumps(cached)

    fetchers = {"pypi": _fetch_pypi, "npm": _fetch_npm, "go": _fetch_go}
    fetcher = fetchers.get(ecosystem, _fetch_pypi)
    result = fetcher(package_name)
    result["is_cached"] = False

    if "error" not in result:
        _write_pkg_cache(package_name, ecosystem, result)

    elapsed = time.monotonic() - t0
    entry = _stats_entry("search_package_info")
    entry["calls"] += 1
    entry["total_s"] += elapsed
    return json.dumps(result)


# ---------------------------------------------------------------------------
# LangChain StructuredTool factory (lazy — only called when deepagents used)
# ---------------------------------------------------------------------------


def build_tools() -> list[Any]:
    """Create LangChain StructuredTool instances wrapping the AIBOM tools.

    This function is called lazily only when ``--agent-model`` is specified,
    so the ``langchain_core`` import happens only when needed.

    ``scan_directory`` is intentionally excluded: the deterministic scan has
    already run and its results are pre-fed in the prompt.  Re-scanning is
    redundant and expensive (~3-4 s per call).
    """
    from langchain_core.tools import StructuredTool

    return [
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
                "context snippets. Use sparingly — only when other tools "
                "cannot answer your question."
            ),
            func=search_codebase_impl,
            args_schema=SearchCodebaseArgs,
        ),
        StructuredTool.from_function(
            name="search_package_info",
            description=(
                "Query a package registry (PyPI, npm, or Go proxy) for "
                "metadata about a dependency. Returns name, summary, "
                "description, keywords, and classifiers. Use this to "
                "determine if a dependency is genuinely AI/ML related."
            ),
            func=search_package_info_impl,
            args_schema=SearchPackageInfoArgs,
        ),
        StructuredTool.from_function(
            name="read_file_snippet",
            description=(
                "Read up to N lines from a file. Use this to inspect the "
                "definition of an imported class, check for agent loop "
                "patterns, or read code that is outside the provided "
                "code_context window. For agent candidates detected via "
                "import, you MUST read the source module to verify the "
                "agent loop pattern before confirming or removing."
            ),
            func=read_file_snippet_impl,
            args_schema=ReadFileSnippetArgs,
        ),
    ]


# ---------------------------------------------------------------------------
# Triage tools — repo exploration for the triage agent
# ---------------------------------------------------------------------------


class ListDirectoryTreeArgs(BaseModel):
    path: str = Field(description="Root directory to list.")
    max_depth: int = Field(default=3, description="Maximum directory depth to recurse.")
    max_entries: int = Field(
        default=200, description="Maximum total entries to return."
    )


@_track_tool("list_directory_tree")
def list_directory_tree_impl(
    path: str, max_depth: int = 3, max_entries: int = 200
) -> str:
    """Recursive directory listing capped by depth and entry count."""
    if _strict_path_is_denied(path):
        _record_guard_denial("list_directory_tree")
        return json.dumps({"error": "path is outside the approved source root"})
    root = Path(path)
    if not root.is_dir():
        return json.dumps({"error": f"not a directory: {path}"})

    entries: list[str] = []
    _SKIP_DIRS = frozenset(
        {
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            ".tox",
            ".mypy_cache",
            ".pytest_cache",
            "dist",
            "build",
            ".eggs",
            "*.egg-info",
        }
    )

    def _walk(cur: Path, depth: int, prefix: str) -> None:
        if depth > max_depth or len(entries) >= max_entries:
            return
        try:
            children = sorted(cur.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        except OSError:
            return
        for child in children:
            if len(entries) >= max_entries:
                entries.append(f"{prefix}... (truncated at {max_entries})")
                return
            if _strict_path_is_denied(str(child)):
                _record_guard_denial("list_directory_tree")
                continue
            if child.is_dir():
                if child.name in _SKIP_DIRS or child.name.endswith(".egg-info"):
                    continue
                entries.append(f"{prefix}{child.name}/")
                _walk(child, depth + 1, prefix + "  ")
            else:
                entries.append(f"{prefix}{child.name}")

    _walk(root, 1, "")
    return "\n".join(entries) if entries else "(empty directory)"


class ReadFileSnippetArgs(BaseModel):
    path: str = Field(description="File path to read.")
    max_lines: int = Field(
        default=200, description="Maximum number of lines to return."
    )


@_track_tool("read_file_snippet")
def read_file_snippet_impl(path: str, max_lines: int = 200) -> str:
    """Read the first N lines of a file."""
    if _strict_path_is_denied(path):
        _record_guard_denial("read_file_snippet")
        return json.dumps({"error": "path is outside the approved source root"})
    p = Path(path)
    if not p.is_file():
        return json.dumps({"error": f"not a file: {path}"})
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        return json.dumps({"error": str(e)})
    truncated = len(lines) > max_lines
    snippet = "\n".join(lines[:max_lines])
    if truncated:
        snippet += f"\n... ({len(lines) - max_lines} more lines)"
    return snippet


def build_triage_tools(repo_root: str) -> list[Any]:
    """Build tools for the repo triage agent.

    Scopes search_codebase to the given repo root.  Bundles directory
    listing, file reading, codebase search, and package-registry lookup.
    """
    from langchain_core.tools import StructuredTool

    bound_root = str(Path(repo_root).resolve())

    def _bound(fn):
        def _invoke(*args, **kwargs):
            token = set_allowed_search_roots([bound_root])
            try:
                return fn(*args, **kwargs)
            finally:
                reset_allowed_search_roots(token)

        _invoke.__name__ = fn.__name__
        _invoke.__doc__ = fn.__doc__
        return _invoke

    return [
        StructuredTool.from_function(
            name="list_directory_tree",
            description=(
                "List the directory tree of a path recursively (up to a depth "
                "limit). Returns file and directory names with indentation."
            ),
            func=_bound(list_directory_tree_impl),
            args_schema=ListDirectoryTreeArgs,
        ),
        StructuredTool.from_function(
            name="read_file_snippet",
            description=(
                "Read the first N lines of a file. Use to inspect README, "
                "manifests, config files, Helm values, or source code."
            ),
            func=_bound(read_file_snippet_impl),
            args_schema=ReadFileSnippetArgs,
        ),
        StructuredTool.from_function(
            name="search_codebase",
            description=(
                "Search across the repository for a regex or literal pattern. "
                "Returns matching file paths, line numbers, and context."
            ),
            func=_bound(search_codebase_impl),
            args_schema=SearchCodebaseArgs,
        ),
        StructuredTool.from_function(
            name="search_package_info",
            description=(
                "Query a package registry (PyPI, npm, or Go proxy) for "
                "metadata about a dependency. Returns name, summary, "
                "description, keywords, and classifiers. Use this to "
                "determine if a dependency is genuinely AI/ML related."
            ),
            func=_bound(search_package_info_impl),
            args_schema=SearchPackageInfoArgs,
        ),
    ]
