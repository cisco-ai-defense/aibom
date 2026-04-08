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

import ast
import re
from pathlib import Path
from typing import Optional

from ..models import AIComponent, ComponentRelationship, ScanContext
from ..models.enums import AIComponentType, DetectionSource
from .base import BaseScanner
from .dependency_scanner import KNOWN_AI_PACKAGES, DependencyScanner, _iter_scan_paths
from .file_cache import is_python_source, read_python_source, read_text_cached

_LANGCHAIN = frozenset({
    "langchain", "langchain-core", "langchain-community", "langchain-openai",
    "langchain-anthropic", "langchain-huggingface", "langchain-chroma",
    "langchain-pinecone", "langchain-google-genai", "langchain-google-vertexai",
    "langchain-cohere", "langchain-mistralai", "langchain-fireworks",
    "langchain-groq", "langchain-together", "langchain-aws",
})
_AUTOGEN = frozenset({"autogen", "autogen-agentchat"})
_LLAMA_INDEX = frozenset({"llama-index", "llama-index-core"})


def _build_module_to_pypi() -> dict[str, frozenset[str]]:
    out: dict[str, frozenset[str]] = {}
    pypi = KNOWN_AI_PACKAGES["pypi"]

    def add(module: str, pkgs: frozenset[str]) -> None:
        out[module] = pkgs

    for mod in (
        "langchain",
        "langchain_core",
        "langchain_community",
        "langchain_openai",
        "langchain_anthropic",
        "langchain_huggingface",
        "langchain_chroma",
        "langchain_pinecone",
        "langchain_google_genai",
        "langchain_google_vertexai",
        "langchain_cohere",
        "langchain_mistralai",
        "langchain_fireworks",
        "langchain_groq",
        "langchain_together",
        "langchain_aws",
    ):
        add(mod, _LANGCHAIN)
    for mod in ("autogen", "autogen_agentchat"):
        add(mod, _AUTOGEN)
    add("tf", frozenset({"tensorflow"}))
    add("google.generativeai", frozenset({"google-generativeai"}))
    add("google.cloud.aiplatform", frozenset({"google-cloud-aiplatform"}))
    add("pinecone", frozenset({"pinecone-client"}))
    add("llama_index", _LLAMA_INDEX)
    add("llama_index.core", frozenset({"llama-index-core"}))
    add("llama_index_core", frozenset({"llama-index-core"}))

    grouped = _LANGCHAIN | _AUTOGEN | _LLAMA_INDEX
    for pkg in pypi:
        if pkg in grouped:
            continue
        if pkg in (
            "google-generativeai",
            "google-cloud-aiplatform",
            "pinecone-client",
        ):
            continue
        root = pkg.replace("-", "_")
        if root in ("llama_index", "llama_index_core"):
            continue
        if root not in out:
            add(root, frozenset({pkg}))
        else:
            prev = out[root]
            out[root] = prev | frozenset({pkg})

    return out


_MODULE_TO_PYPI = _build_module_to_pypi()

_IMPORTLIB_MOD = re.compile(
    r"""importlib\.import_module\s*\(\s*["']([^"']+)["']""",
)
_IMPORT_BUILTIN_RE = re.compile(r"""__import__\s*\(\s*["']([^"']+)["']""")


def _pypi_display_name(spec: str, pkgs: frozenset[str]) -> str:
    spec = spec.strip()
    if len(pkgs) == 1:
        return next(iter(pkgs))
    if pkgs == _LANGCHAIN:
        root = spec.split(".")[0]
        inferred = root.replace("_", "-")
        if inferred in pkgs:
            return inferred
        return "langchain"
    if pkgs == _AUTOGEN:
        root = spec.split(".")[0]
        inferred = root.replace("_", "-")
        if inferred in pkgs:
            return inferred
        return "autogen"
    if pkgs == _LLAMA_INDEX:
        root = spec.split(".")[0]
        if root == "llama_index_core" or ".core" in spec:
            return "llama-index-core"
        return "llama-index"
    return min(pkgs)


def _satisfying_pypi_packages(module_spec: str) -> Optional[frozenset[str]]:
    s = module_spec.strip()
    if s in _MODULE_TO_PYPI:
        return _MODULE_TO_PYPI[s]
    for prefix, pkgs in (
        ("google.generativeai", _MODULE_TO_PYPI["google.generativeai"]),
        ("google.cloud.aiplatform", _MODULE_TO_PYPI["google.cloud.aiplatform"]),
    ):
        if s == prefix or s.startswith(prefix + "."):
            return pkgs
    root = s.split(".")[0]
    if root == "tf":
        return _MODULE_TO_PYPI.get("tf")
    return _MODULE_TO_PYPI.get(root)


def _is_declared(declared: set[str], pkgs: frozenset[str]) -> bool:
    return bool(declared.intersection(pkgs))


def _string_from_expr(node: ast.expr) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


class _ImportCollector(ast.NodeVisitor):
    def __init__(self, line_texts: list[str]) -> None:
        self._lines = line_texts
        self.hits: list[tuple[int, str, str]] = []

    def _record(self, lineno: int, module_spec: str, snippet: str) -> None:
        pkgs = _satisfying_pypi_packages(module_spec)
        if pkgs is None:
            return
        self.hits.append((lineno, module_spec, snippet))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            spec = alias.name
            line = self._lines[node.lineno - 1] if node.lineno else ""
            self._record(node.lineno, spec, line.strip())
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level and node.level > 0:
            self.generic_visit(node)
            return
        if not node.module:
            self.generic_visit(node)
            return
        line = self._lines[node.lineno - 1] if node.lineno else ""
        snippet = line.strip()
        for alias in node.names:
            if alias.name == "*":
                continue
            cand = f"{node.module}.{alias.name}"
            if _satisfying_pypi_packages(cand) is not None:
                self._record(node.lineno, cand, snippet)
            elif _satisfying_pypi_packages(node.module) is not None:
                self._record(node.lineno, node.module, snippet)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        fn = node.func
        target: Optional[str] = None
        if isinstance(fn, ast.Attribute):
            if (
                isinstance(fn.value, ast.Name)
                and fn.value.id == "importlib"
                and fn.attr == "import_module"
                and node.args
            ):
                target = _string_from_expr(node.args[0])
        elif isinstance(fn, ast.Name) and fn.id == "__import__" and node.args:
            target = _string_from_expr(node.args[0])
        if target is not None and node.lineno:
            line = self._lines[node.lineno - 1]
            self._record(node.lineno, target, line.strip())
        self.generic_visit(node)


def _regex_dynamic_hits(
    text: str, line_texts: list[str]
) -> list[tuple[int, str, str]]:
    hits: list[tuple[int, str, str]] = []
    for i, line in enumerate(line_texts, start=1):
        for pat in (_IMPORTLIB_MOD, _IMPORT_BUILTIN_RE):
            for m in pat.finditer(line):
                spec = m.group(1)
                if _satisfying_pypi_packages(spec) is not None:
                    hits.append((i, spec, line.strip()))
    return hits


def _collect_py_imports(path: Path) -> list[tuple[int, str, str]]:
    try:
        raw = read_python_source(path)
    except OSError:
        return []
    lines = raw.splitlines()
    try:
        tree = ast.parse(raw, filename=str(path))
    except SyntaxError:
        tree = None
    hits: list[tuple[int, str, str]] = []
    if tree is not None:
        col = _ImportCollector(lines)
        col.visit(tree)
        hits.extend(col.hits)
    hits.extend(_regex_dynamic_hits(raw, lines))
    seen: set[tuple[int, str]] = set()
    out: list[tuple[int, str, str]] = []
    for lineno, spec, snip in hits:
        key = (lineno, spec)
        if key in seen:
            continue
        seen.add(key)
        out.append((lineno, spec, snip))
    return out


class ShadowAIDetector(BaseScanner):
    name = "shadow_ai_detector"

    def supports(self, context: ScanContext) -> bool:
        return True

    def scan(
        self, context: ScanContext
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        dep = DependencyScanner()
        declared_components, _ = dep.scan(context)
        declared: set[str] = set()
        for c in declared_components:
            if c.metadata.get("ecosystem") == "pypi":
                declared.add(c.name.strip().lower())

        findings: list[AIComponent] = []
        seen: set[tuple[str, str]] = set()
        for path in _iter_scan_paths(context):
            if not is_python_source(path):
                continue
            for lineno, mod_spec, snippet in _collect_py_imports(path):
                pkgs = _satisfying_pypi_packages(mod_spec)
                if pkgs is None:
                    continue
                if _is_declared(declared, pkgs):
                    continue
                pkg_name = _pypi_display_name(mod_spec, pkgs)
                key = (str(path.resolve()), pkg_name)
                if key in seen:
                    continue
                seen.add(key)
                findings.append(
                    AIComponent(
                        name=pkg_name,
                        component_type=AIComponentType.DEPENDENCY,
                        file_path=str(path.resolve()),
                        line_number=lineno,
                        detection_source=DetectionSource.CODE_ANALYSIS,
                        description=(
                            f"Undeclared AI dependency: {pkg_name} imported in code "
                            "but not in dependency manifests"
                        ),
                        metadata={
                            "shadow_ai": True,
                            "import_found": snippet,
                            "declared_in_manifests": False,
                        },
                    ),
                )
        return findings, []
