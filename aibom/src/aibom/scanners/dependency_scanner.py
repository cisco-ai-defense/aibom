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
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import yaml  # type: ignore[import-untyped]
from pathspec import PathSpec

from ..models import AIComponent, ComponentRelationship, ScanContext
from ..models.enums import AIComponentType, DetectionSource
from .base import BaseScanner

KNOWN_AI_PACKAGES: dict[str, set[str]] = {
    "pypi": {
        "openai",
        "langchain",
        "langchain-core",
        "langchain-community",
        "langchain-openai",
        "langchain-anthropic",
        "langchain-huggingface",
        "langchain-chroma",
        "langchain-pinecone",
        "langchain-google-genai",
        "langchain-google-vertexai",
        "langchain-cohere",
        "langchain-mistralai",
        "langchain-fireworks",
        "langchain-groq",
        "langchain-together",
        "langchain-aws",
        "anthropic",
        "google-generativeai",
        "google-cloud-aiplatform",
        "crewai",
        "autogen",
        "autogen-agentchat",
        "mcp",
        "fastmcp",
        "llama-index",
        "llama-index-core",
        "chromadb",
        "pinecone-client",
        "weaviate-client",
        "qdrant-client",
        "transformers",
        "torch",
        "tensorflow",
        "keras",
        "jax",
        "flax",
        "sentence-transformers",
        "diffusers",
        "accelerate",
        "peft",
        "trl",
        "datasets",
        "evaluate",
        "tokenizers",
        "safetensors",
        "bitsandbytes",
        "vllm",
        "text-generation-inference",
        "mlflow",
        "wandb",
        "dvc",
        "comet-ml",
        "neptune",
        "ray",
        "kubeflow",
        "bentoml",
        "litellm",
        "dspy",
        "instructor",
        "guidance",
        "semantic-kernel",
        "promptflow",
        "deepeval",
        "google-genai",
        "guardrails",
        "llmetry",
        "openai-agents",
        # Observability
        "traceloop-sdk",
        "openllmetry",
        "freeplay",
        "langsmith",
        "langfuse",
        "arize-phoenix",
        "opik",
        "helicone",
        "tracia",
        "opentelemetry-instrumentation-anthropic",
        "opentelemetry-instrumentation-chromadb",
        "opentelemetry-instrumentation-crewai",
        "opentelemetry-instrumentation-google-generativeai",
        "opentelemetry-instrumentation-llamaindex",
        "opentelemetry-instrumentation-mistralai",
        "opentelemetry-instrumentation-openai",
        "opentelemetry-instrumentation-openai-v2",
        "opentelemetry-instrumentation-vertexai",
        "opentelemetry-instrumentation-weaviate",
        "opentelemetry-instrumentation-langchain",
        "opentelemetry-semantic-conventions-ai",
        # Guardrails
        "nemoguardrails",
        "guardrails-ai",
        "llm-guard",
        "lakera-guard",
        "rebuff",
        "cisco-aidefense-sdk",
        # MCP clients
        "mcp-client",
        # AWS Strands agent framework
        "strands-agents",
        "strands-agents-tools",
        "mcp-proxy-for-aws",
    },
    "npm": {
        "openai",
        "@langchain/core",
        "@langchain/openai",
        "@langchain/anthropic",
        "@langchain/community",
        "@anthropic-ai/sdk",
        "ai",
        "@ai-sdk/openai",
        "@google/generative-ai",
        "cohere-ai",
        "@modelcontextprotocol/sdk",
        "llamaindex",
        "chromadb",
        "@pinecone-database/pinecone",
        "weaviate-ts-client",
        "@qdrant/js-client-rest",
        "langsmith",
        "langfuse",
    },
    "go": {
        "github.com/sashabaranov/go-openai",
        "github.com/anthropics/anthropic-sdk-go",
        "github.com/google/generative-ai-go",
        "github.com/tmc/langchaingo",
    },
    "maven": {
        "dev.langchain4j",
        "com.azure:azure-ai-openai",
        "com.google.cloud:google-cloud-aiplatform",
        "io.github.sashirestela:simple-openai",
    },
    "cargo": {
        "async-openai",
        "llm",
        "candle-core",
        "candle-nn",
        "burn",
    },
    "rubygems": {
        "ruby-openai",
        "langchainrb",
    },
}

_MAVEN_GROUP_PREFIX = "dev.langchain4j"
_MAVEN_EXACT_COORDS = frozenset(
    c
    for c in KNOWN_AI_PACKAGES["maven"]
    if ":" in c and not c.startswith(_MAVEN_GROUP_PREFIX + ":")
)

_REQUIREMENT_NAME = re.compile(r"^([a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?)")
_PIPFILE_PKG = re.compile(r"^([a-zA-Z0-9_.-]+)\s*=")
_TOML_STRING_DEP = re.compile(r"""["']([^"']+)["']""")
_POETRY_DEP_KEY = re.compile(r"^([a-zA-Z0-9_.-]+)\s*=")
_PURE_VERSION_TOKEN = re.compile(r"^[vV]?\d+(?:\.\d+)+(?:[a-zA-Z0-9._-]*)?$")
_LOCK_PACKAGE = re.compile(r"^\[\[package\]\]\s*$")
_LOCK_NAME = re.compile(r'^name\s*=\s*["\']([^"\']+)["\']')
_LOCK_VERSION = re.compile(r'^version\s*=\s*["\']([^"\']+)["\']')
_INSTALL_REQUIRES = re.compile(r"install_requires\s*=\s*\[([\s\S]*?)\]", re.MULTILINE)
_GRADLE_DEP = re.compile(
    r"""(?:(?:implementation|api|compileOnly|runtimeOnly|compile|classpath)\s*)"""
    r"""[\(\s]*["']([^"']+)["']""",
    re.MULTILINE,
)
_GRADLE_KTS = re.compile(
    r"""(?:implementation|api|compileOnly|runtimeOnly)\s*\(\s*["']([^"']+)["']\s*\)""",
    re.MULTILINE,
)
_POM_DEP = re.compile(
    r"<dependency>\s*([\s\S]*?)</dependency>", re.IGNORECASE | re.DOTALL
)
_POM_GROUP = re.compile(r"<groupId>\s*([^<]+?)\s*</groupId>", re.IGNORECASE)
_POM_ARTIFACT = re.compile(r"<artifactId>\s*([^<]+?)\s*</artifactId>", re.IGNORECASE)
_POM_VERSION = re.compile(r"<version>\s*([^<]+?)\s*</version>", re.IGNORECASE)
_CSPROJ_REF = re.compile(
    r'<PackageReference\s+Include\s*=\s*["\']([^"\']+)["\']\s+'
    r'Version\s*=\s*["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_GO_REQUIRE = re.compile(
    r"^([a-zA-Z0-9_.\-/]+(?:/v\d+)?)\s+v([0-9][^\s#]*)",
)
_GEM_LINE = re.compile(
    r"""gem\s+["']([^"']+)["']""" r"""(?:\s*,\s*["']([^"']*)["'])?""",
)
_SETUP_CFG_REQUIRES = re.compile(r"(?m)^install_requires\s*=\s*\n((?:[ \t]+[^\n]+\n)+)")


def _normalize_pypi_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _pypi_spec_pinned(raw: str) -> tuple[bool, Optional[str]]:
    s = raw.strip()
    if "===" in s:
        parts = s.split("===", 1)
        return True, parts[1].strip() if len(parts) > 1 else None
    if "==" in s:
        parts = s.split("==", 1)
        return True, parts[1].strip() if len(parts) > 1 else None
    if s and re.fullmatch(r"[0-9][0-9a-zA-Z._\-*!]*", s):
        return True, s
    return False, None


def _parse_pep508_name_version(spec: str) -> tuple[str, str, bool, Optional[str]]:
    spec = spec.strip()
    if not spec or spec.startswith("#"):
        return "", "", False, None
    base = spec.split(";", 1)[0].strip()
    sep_used: Optional[str] = None
    name_part = base
    ver_part = ""
    for sep in ("===", "==", ">=", "<=", "~=", "!=", ">", "<"):
        if sep in base:
            name_part, ver_part = base.split(sep, 1)
            name_part = name_part.strip()
            ver_part = ver_part.strip()
            sep_used = sep
            break
    if "[" in name_part:
        name_part = name_part.split("[", 1)[0].strip()
    m = _REQUIREMENT_NAME.match(name_part)
    if not m:
        return "", "", False, None
    name = m.group(1)
    if _PURE_VERSION_TOKEN.fullmatch(name):
        return "", "", False, None
    if not ver_part:
        return name, "*", False, None
    if sep_used in ("==", "==="):
        pinned, v = _pypi_spec_pinned(ver_part)
        return name, ver_part, pinned, v
    return name, ver_part, False, None


def _parse_requirements_txt(
    text: str, base_dir: Path, visited: Optional[set[Path]] = None
) -> list[tuple[str, str, int, Optional[str], str]]:
    out: list[tuple[str, str, int, Optional[str], str]] = []
    if visited is None:
        visited = set()
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        lineno = i + 1
        i += 1
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("-r ") or stripped.startswith("--requirement "):
            inc = stripped.split(None, 1)[1].strip()
            p = (base_dir / inc).resolve()
            if p in visited or not p.is_file():
                continue
            visited.add(p)
            try:
                sub = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            out.extend(_parse_requirements_txt(sub, p.parent, visited))
            continue
        name, raw_spec, pinned, ver = _parse_pep508_name_version(stripped)
        if not name:
            continue
        out.append((name, raw_spec, lineno, ver if pinned else None, "pypi"))
    return out


def _section_lines(text: str, header: str) -> list[str]:
    header_l = header.lower()
    in_section = False
    collected: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            in_section = s.lower() == header_l
            continue
        if in_section:
            if s.startswith("[") and s.endswith("]"):
                break
            collected.append(line)
    return collected


def _parse_pipfile(text: str) -> list[tuple[str, str, int, Optional[str], str]]:
    out: list[tuple[str, str, int, Optional[str], str]] = []
    lines = _section_lines(text, "[packages]")
    for idx, line in enumerate(lines, start=1):
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        m = _PIPFILE_PKG.match(raw)
        if not m:
            continue
        name = m.group(1)
        rest = raw[m.end() :].strip()
        ver_m = re.search(r'version\s*=\s*["\']([^"\']+)["\']', rest)
        quoted = re.search(r"=\s*[\"']([^\"']+)[\"']", rest)
        raw_spec = ver_m.group(1) if ver_m else (quoted.group(1) if quoted else "*")
        pinned, pv = _pypi_spec_pinned(raw_spec)
        out.append((name, raw_spec, idx, pv if pinned else None, "pypi"))
    return out


def _parse_pyproject_toml(text: str) -> list[tuple[str, str, int, Optional[str], str]]:
    out: list[tuple[str, str, int, Optional[str], str]] = []
    lower = text.lower()
    for marker in ("[project.dependencies]",):
        pos = lower.find(marker.lower())
        if pos < 0:
            continue
        chunk = text[pos:]
        end = re.search(r"^\[", chunk[1:], re.MULTILINE)
        section = chunk[: end.start() + 1] if end else chunk
        line_base = text[:pos].count("\n")
        for i, line in enumerate(section.splitlines(), start=0):
            ls = line.strip()
            if marker.endswith("poetry.dependencies") and ls.startswith("python "):
                continue
            for m in _TOML_STRING_DEP.finditer(line):
                spec = m.group(1)
                name, raw_spec, pinned, ver = _parse_pep508_name_version(spec)
                if name:
                    out.append(
                        (
                            name,
                            raw_spec,
                            line_base + i + 1,
                            ver if pinned else None,
                            "pypi",
                        ),
                    )
    poetry_re = re.compile(r"^\[(tool\.poetry(?:\.group\.[^.]+)?\.dependencies)\]\s*$", re.MULTILINE | re.IGNORECASE)
    for m in poetry_re.finditer(text):
        start = m.end()
        rest = text[start:]
        endm = re.search(r"^\[", rest, re.MULTILINE)
        block = rest[: endm.start()] if endm else rest
        line_base = text[:start].count("\n")
        for i, line in enumerate(block.splitlines(), start=1):
            ls = line.strip()
            if not ls or ls.startswith("#"):
                continue
            key_match = _POETRY_DEP_KEY.match(ls)
            if not key_match:
                continue
            name = key_match.group(1)
            if name == "python" or _PURE_VERSION_TOKEN.fullmatch(name):
                continue
            rest = ls[key_match.end() :].strip()
            ver_m = re.search(r'version\s*=\s*["\']([^"\']+)["\']', rest)
            quoted = re.search(r'["\']([^"\']+)["\']', rest)
            raw_spec = ver_m.group(1) if ver_m else (quoted.group(1) if quoted else "*")
            pinned, ver = _pypi_spec_pinned(raw_spec)
            out.append(
                (
                    name,
                    raw_spec,
                    line_base + i,
                    ver if pinned else None,
                    "pypi",
                ),
            )
    opt_re = re.compile(
        r"^\[project\.optional-dependencies\.([^\]]+)\]\s*$",
        re.MULTILINE | re.IGNORECASE,
    )
    for m in opt_re.finditer(text):
        start = m.end()
        rest = text[start:]
        endm = re.search(r"^\[", rest, re.MULTILINE)
        block = rest[: endm.start()] if endm else rest
        line_base = text[:start].count("\n")
        for i, line in enumerate(block.splitlines(), start=1):
            for sm in _TOML_STRING_DEP.finditer(line):
                spec = sm.group(1)
                name, raw_spec, pinned, ver = _parse_pep508_name_version(spec)
                if name:
                    out.append(
                        (
                            name,
                            raw_spec,
                            line_base + i,
                            ver if pinned else None,
                            "pypi",
                        ),
                    )
    return out


def _parse_setup_content(text: str) -> list[tuple[str, str, int, Optional[str], str]]:
    out: list[tuple[str, str, int, Optional[str], str]] = []
    for m in _INSTALL_REQUIRES.finditer(text):
        inner = m.group(1)
        line_base = text[: m.start()].count("\n")
        for sm in _TOML_STRING_DEP.finditer(inner):
            spec = sm.group(1)
            name, raw_spec, pinned, ver = _parse_pep508_name_version(spec)
            if name:
                rel_line = inner[: sm.start()].count("\n")
                out.append(
                    (
                        name,
                        raw_spec,
                        line_base + rel_line + 1,
                        ver if pinned else None,
                        "pypi",
                    ),
                )
    for m in re.finditer(r"install_requires\s*=\s*\(([\s\S]*?)\)", text, re.MULTILINE):
        inner = m.group(1)
        line_base = text[: m.start()].count("\n")
        for sm in _TOML_STRING_DEP.finditer(inner):
            spec = sm.group(1)
            name, raw_spec, pinned, ver = _parse_pep508_name_version(spec)
            if name:
                rel_line = inner[: sm.start()].count("\n")
                out.append(
                    (
                        name,
                        raw_spec,
                        line_base + rel_line + 1,
                        ver if pinned else None,
                        "pypi",
                    ),
                )
    return out


def _parse_setup_cfg(text: str) -> list[tuple[str, str, int, Optional[str], str]]:
    out: list[tuple[str, str, int, Optional[str], str]] = []
    for m in _SETUP_CFG_REQUIRES.finditer(text):
        block = m.group(1)
        line_base = text[: m.start(1)].count("\n") + 1
        for i, line in enumerate(block.splitlines()):
            spec = line.strip()
            if not spec or spec.startswith("#"):
                continue
            name, raw_spec, pinned, ver = _parse_pep508_name_version(spec)
            if name:
                out.append(
                    (name, raw_spec, line_base + i, ver if pinned else None, "pypi"),
                )
    joined = "\n".join(_section_lines(text, "[options]"))
    out.extend(_parse_setup_content(joined))
    return out


def _parse_poetry_uv_lock(text: str) -> list[tuple[str, str, int, Optional[str], str]]:
    out: list[tuple[str, str, int, Optional[str], str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if _LOCK_PACKAGE.match(lines[i].strip()):
            block_start = i + 1
            i += 1
            name_v: Optional[str] = None
            ver_v: Optional[str] = None
            name_line = block_start
            while i < len(lines) and not lines[i].strip().startswith("["):
                ln = lines[i]
                nm = _LOCK_NAME.match(ln.strip())
                if nm:
                    name_v = nm.group(1)
                    name_line = i + 1
                vm = _LOCK_VERSION.match(ln.strip())
                if vm:
                    ver_v = vm.group(1)
                i += 1
            if name_v:
                out.append((name_v, ver_v or "", name_line, ver_v, "pypi"))
        else:
            i += 1
    return out


def _line_for_needle(text: str, needle: str) -> int:
    for idx, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return idx
    return 1


def _npm_pinned_version(raw: str) -> Optional[str]:
    v = raw.strip()
    if not v or v[0] in "^~>*<":
        return None
    if "||" in v or " - " in v:
        return None
    if re.fullmatch(r"\d[\d.]*", v):
        return v
    if re.fullmatch(r"\d+\.\d+\.\d+[\w.-]*", v):
        return v
    return None


def _parse_package_json(text: str) -> list[tuple[str, str, int, Optional[str], str]]:
    out: list[tuple[str, str, int, Optional[str], str]] = []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return out
    for key in ("dependencies", "devDependencies"):
        block = data.get(key)
        if not isinstance(block, dict):
            continue
        for name, ver in block.items():
            if not isinstance(ver, str):
                continue
            raw = ver.strip()
            pv = _npm_pinned_version(raw)
            needle = f'"{name}"'
            ln = _line_for_needle(text, needle)
            out.append((name, raw, ln, pv, "npm"))
    return out


def _walk_pkg_lock_deps(
    obj: object,
    text: str,
    out: list[tuple[str, str, int, Optional[str], str]],
) -> None:
    if not isinstance(obj, dict):
        return
    for name, meta in obj.items():
        if not isinstance(meta, dict):
            continue
        ver = meta.get("version")
        if isinstance(ver, str):
            ln = _line_for_needle(text, f'"{name}"')
            out.append((name, ver, ln, ver, "npm"))
        nested = meta.get("dependencies")
        if isinstance(nested, dict):
            _walk_pkg_lock_deps(nested, text, out)


def _parse_package_lock_json(
    text: str,
) -> list[tuple[str, str, int, Optional[str], str]]:
    out: list[tuple[str, str, int, Optional[str], str]] = []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return out
    packages = data.get("packages")
    if isinstance(packages, dict) and len(packages) > 0:
        for path_key, meta in packages.items():
            if not isinstance(meta, dict):
                continue
            if path_key == "":
                continue
            ver = meta.get("version")
            if not isinstance(ver, str):
                continue
            name = meta.get("name")
            if isinstance(name, str) and name:
                disp = name
            else:
                parts = path_key.replace("\\", "/").split("/")
                if "node_modules" in parts:
                    idx = len(parts) - 1
                    while idx >= 0 and parts[idx] != "node_modules":
                        idx -= 1
                    disp = "/".join(parts[idx + 1 :]) if idx >= 0 else path_key
                else:
                    disp = path_key
            needle = disp.split("/")[-1]
            ln = _line_for_needle(text, needle)
            out.append((disp, ver, ln, ver, "npm"))
    else:
        deps = data.get("dependencies")
        if isinstance(deps, dict):
            _walk_pkg_lock_deps(deps, text, out)
    return out


def _parse_yarn_lock(text: str) -> list[tuple[str, str, int, Optional[str], str]]:
    out: list[tuple[str, str, int, Optional[str], str]] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        if line.startswith("#") or not line.strip():
            continue
        if re.match(r"^[\w@./-]+@[^:]+:\s*$", line.strip()):
            head = line.strip().rstrip(":").strip('"')
            at_idx = head.rfind("@")
            if at_idx <= 0:
                continue
            name = head[:at_idx]
            ver = head[at_idx + 1 :]
            if name.startswith('"') and name.endswith('"'):
                name = name[1:-1]
            out.append((name, ver, idx, ver, "npm"))
    return out


def _parse_pnpm_lock_yaml(text: str) -> list[tuple[str, str, int, Optional[str], str]]:
    out: list[tuple[str, str, int, Optional[str], str]] = []
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return out
    packages = data.get("packages") if isinstance(data, dict) else None
    if not isinstance(packages, dict):
        return out
    for key, _ in packages.items():
        if not isinstance(key, str):
            continue
        k = key.strip("/")
        parts = k.split("/")
        if len(parts) >= 3 and parts[0] == "" and parts[1].startswith("@"):
            name = f"{parts[1]}/{parts[2]}"
            ver = parts[3] if len(parts) > 3 else ""
        elif len(parts) >= 2 and parts[0] == "":
            name, ver = parts[1], parts[2] if len(parts) > 2 else ""
        elif len(parts) >= 2 and parts[0].startswith("@"):
            name = f"{parts[0]}/{parts[1]}"
            ver = parts[2] if len(parts) > 2 else ""
        elif len(parts) >= 2:
            name, ver = parts[0], parts[1]
        else:
            continue
        pv: Optional[str] = ver if ver and ver[0].isdigit() else None
        ln = _line_for_needle(text, k[: min(60, len(k))])
        out.append((name, ver or "*", ln, pv, "npm"))
    return out


def _parse_cargo_toml(text: str) -> list[tuple[str, str, int, Optional[str], str]]:
    out: list[tuple[str, str, int, Optional[str], str]] = []
    lines = _section_lines(text, "[dependencies]")
    for idx, line in enumerate(lines, start=1):
        raw = line.split("#", 1)[0].strip()
        if not raw:
            continue
        m = _POETRY_DEP_KEY.match(raw)
        if not m:
            continue
        name = m.group(1)
        rest = raw[m.end() :].strip()
        ver_m = re.search(r'version\s*=\s*["\']([^"\']+)["\']', rest)
        simple = re.search(r"=\s*[\"']([^\"']+)[\"']", rest)
        raw_spec = ver_m.group(1) if ver_m else (simple.group(1) if simple else "*")
        pv: Optional[str] = None
        if ver_m:
            pv = ver_m.group(1)
        elif simple and re.fullmatch(r"[\d.]+", simple.group(1)):
            pv = simple.group(1)
        out.append((name, raw_spec, idx, pv, "cargo"))
    return out


def _parse_go_mod(text: str) -> list[tuple[str, str, int, Optional[str], str]]:
    out: list[tuple[str, str, int, Optional[str], str]] = []
    in_require = False
    for idx, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if s.startswith("require ("):
            in_require = True
            continue
        if in_require:
            if s == ")":
                in_require = False
                continue
            m = _GO_REQUIRE.match(line.strip())
            if m:
                out.append((m.group(1), m.group(2), idx, m.group(2), "go"))
            continue
        if s.startswith("require ") and "(" not in s:
            rest = s[8:].strip()
            m = _GO_REQUIRE.match(rest)
            if m:
                out.append((m.group(1), m.group(2), idx, m.group(2), "go"))
    return out


def _parse_gemfile(text: str) -> list[tuple[str, str, int, Optional[str], str]]:
    out: list[tuple[str, str, int, Optional[str], str]] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = _GEM_LINE.match(s)
        if m:
            name = m.group(1)
            ver = (m.group(2) or "").strip()
            pv = ver if ver and re.fullmatch(r"[\d.]+", ver) else None
            out.append((name, ver or "*", idx, pv, "rubygems"))
    return out


def _parse_pom_xml(text: str) -> list[tuple[str, str, int, Optional[str], str]]:
    out: list[tuple[str, str, int, Optional[str], str]] = []
    for dm in _POM_DEP.finditer(text):
        block = dm.group(1)
        gm = _POM_GROUP.search(block)
        am = _POM_ARTIFACT.search(block)
        if not gm or not am:
            continue
        g, a = gm.group(1).strip(), am.group(1).strip()
        coord = f"{g}:{a}"
        vm = _POM_VERSION.search(block)
        ver_raw = vm.group(1).strip() if vm else ""
        if ver_raw.startswith("${"):
            pv: Optional[str] = None
        else:
            pv = ver_raw if ver_raw else None
        line = text[: dm.start()].count("\n") + 1
        out.append((coord, ver_raw or "*", line, pv, "maven"))
    return out


def _parse_gradle(text: str) -> list[tuple[str, str, int, Optional[str], str]]:
    out: list[tuple[str, str, int, Optional[str], str]] = []
    for m in _GRADLE_DEP.finditer(text):
        spec = m.group(1)
        parts = spec.split(":")
        line = text[: m.start()].count("\n") + 1
        if len(parts) >= 3:
            g, a, v = parts[0], parts[1], ":".join(parts[2:])
            coord = f"{g}:{a}"
            pv = v if v and not v.startswith("$") else None
            out.append((coord, v, line, pv, "maven"))
        elif len(parts) == 2:
            out.append((f"{parts[0]}:{parts[1]}", "*", line, None, "maven"))
    for m in _GRADLE_KTS.finditer(text):
        spec = m.group(1)
        parts = spec.split(":")
        line = text[: m.start()].count("\n") + 1
        if len(parts) >= 3:
            g, a, v = parts[0], parts[1], ":".join(parts[2:])
            coord = f"{g}:{a}"
            pv = v if v and not v.startswith("$") else None
            out.append((coord, v, line, pv, "maven"))
    return out


def _parse_csproj(text: str) -> list[tuple[str, str, int, Optional[str], str]]:
    out: list[tuple[str, str, int, Optional[str], str]] = []
    for m in _CSPROJ_REF.finditer(text):
        name, ver = m.group(1), m.group(2)
        line = text[: m.start()].count("\n") + 1
        pv = ver if ver and not ver.startswith("$") else None
        out.append((name, ver, line, pv, "nuget"))
    return out


def _parse_requirements_file(
    path: Path, text: str
) -> list[tuple[str, str, int, Optional[str], str]]:
    return _parse_requirements_txt(text, path.parent)


def _parse_pipfile_file(
    path: Path, text: str
) -> list[tuple[str, str, int, Optional[str], str]]:
    return _parse_pipfile(text)


def _parse_pyproject_file(
    path: Path, text: str
) -> list[tuple[str, str, int, Optional[str], str]]:
    return _parse_pyproject_toml(text)


def _parse_setup_py_file(
    path: Path, text: str
) -> list[tuple[str, str, int, Optional[str], str]]:
    return _parse_setup_content(text)


def _parse_setup_cfg_file(
    path: Path, text: str
) -> list[tuple[str, str, int, Optional[str], str]]:
    return _parse_setup_cfg(text)


def _parse_lock_file(
    path: Path, text: str
) -> list[tuple[str, str, int, Optional[str], str]]:
    return _parse_poetry_uv_lock(text)


def _parse_package_json_file(
    path: Path, text: str
) -> list[tuple[str, str, int, Optional[str], str]]:
    return _parse_package_json(text)


def _parse_package_lock_file(
    path: Path, text: str
) -> list[tuple[str, str, int, Optional[str], str]]:
    return _parse_package_lock_json(text)


def _parse_yarn_lock_file(
    path: Path, text: str
) -> list[tuple[str, str, int, Optional[str], str]]:
    return _parse_yarn_lock(text)


def _parse_pnpm_file(
    path: Path, text: str
) -> list[tuple[str, str, int, Optional[str], str]]:
    return _parse_pnpm_lock_yaml(text)


def _parse_cargo_file(
    path: Path, text: str
) -> list[tuple[str, str, int, Optional[str], str]]:
    return _parse_cargo_toml(text)


def _parse_go_mod_file(
    path: Path, text: str
) -> list[tuple[str, str, int, Optional[str], str]]:
    return _parse_go_mod(text)


def _parse_gemfile_file(
    path: Path, text: str
) -> list[tuple[str, str, int, Optional[str], str]]:
    return _parse_gemfile(text)


def _parse_pom_file(
    path: Path, text: str
) -> list[tuple[str, str, int, Optional[str], str]]:
    return _parse_pom_xml(text)


def _parse_gradle_file(
    path: Path, text: str
) -> list[tuple[str, str, int, Optional[str], str]]:
    return _parse_gradle(text)


def _parse_gradle_kts_file(
    path: Path, text: str
) -> list[tuple[str, str, int, Optional[str], str]]:
    return _parse_gradle(text)


_ManifestParser = Callable[[Path, str], list[tuple[str, str, int, Optional[str], str]]]

_MANIFEST_PARSERS: dict[str, _ManifestParser] = {
    "requirements.txt": _parse_requirements_file,
    "pipfile": _parse_pipfile_file,
    "pyproject.toml": _parse_pyproject_file,
    "setup.py": _parse_setup_py_file,
    "setup.cfg": _parse_setup_cfg_file,
    "poetry.lock": _parse_lock_file,
    "uv.lock": _parse_lock_file,
    "package.json": _parse_package_json_file,
    "package-lock.json": _parse_package_lock_file,
    "yarn.lock": _parse_yarn_lock_file,
    "pnpm-lock.yaml": _parse_pnpm_file,
    "cargo.toml": _parse_cargo_file,
    "go.mod": _parse_go_mod_file,
    "gemfile": _parse_gemfile_file,
    "pom.xml": _parse_pom_file,
    "build.gradle": _parse_gradle_file,
    "build.gradle.kts": _parse_gradle_kts_file,
}


def _is_known_ai(ecosystem: str, name: str) -> bool:
    """Check if a package is in the known-AI hint list (not a gate)."""
    if ecosystem == "nuget":
        return False
    if ecosystem == "maven":
        g, _, a = name.partition(":")
        if not a:
            return False
        coord = f"{g}:{a}"
        if coord in _MAVEN_EXACT_COORDS:
            return True
        if g == _MAVEN_GROUP_PREFIX or g.startswith(_MAVEN_GROUP_PREFIX + "."):
            return True
        return False
    if ecosystem == "go":
        for mod in KNOWN_AI_PACKAGES["go"]:
            if name == mod or name.startswith(mod + "/"):
                return True
        return False
    pkgs = KNOWN_AI_PACKAGES.get(ecosystem, set())
    if ecosystem == "pypi":
        return _normalize_pypi_name(name) in pkgs
    return name in pkgs


def is_known_ai_package(ecosystem: str, name: str) -> bool:
    """Public helper for enforcing the AI-only dependency policy."""
    return _is_known_ai(ecosystem, name)


def _display_name(ecosystem: str, name: str) -> str:
    if ecosystem == "pypi":
        return _normalize_pypi_name(name)
    return name


def _iter_scan_paths(context: ScanContext) -> Iterator[Path]:
    from ..utils.path_filter import should_skip_dir

    spec: Optional[PathSpec] = None
    if context.exclude_patterns:
        spec = PathSpec.from_lines("gitwildmatch", context.exclude_patterns)
    for p in context.paths:
        root = Path(p)
        if not root.exists():
            continue
        if root.is_file():
            if spec:
                rel = root.name
                if spec.match_file(rel):
                    continue
            yield root
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames if not should_skip_dir(d)
            ]
            for fname in filenames:
                f = Path(dirpath) / fname
                try:
                    rel = f.relative_to(root).as_posix()
                except ValueError:
                    rel = str(f)
                if spec and spec.match_file(rel):
                    continue
                yield f


def _parse_manifest(path: Path) -> list[tuple[str, str, int, Optional[str], str]]:
    name = path.name.lower()
    if name.endswith(".csproj"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return _parse_csproj(text)
    parser = _MANIFEST_PARSERS.get(name)
    if not parser:
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return parser(path, text)


def discover_ai_package_set(context: ScanContext) -> frozenset[str]:
    """Run a fast manifest-only pass and return the set of AI package names.

    This is Pass 1 of the two-pass pipeline: structural parsing of manifests
    (requirements.txt, pyproject.toml, poetry.lock, package.json, go.mod, etc.)
    to discover which AI packages the project actually uses.  The result is
    used to scope Pass 2 scanners so they only deep-scan relevant files.

    Package names are normalised (lowered, hyphens/underscores unified for PyPI).
    """
    pkgs: set[str] = set()
    for path in _iter_scan_paths(context):
        key = path.name.lower()
        if key not in _MANIFEST_PARSERS and not key.endswith(".csproj"):
            continue
        for name, _spec, _line, _ver, ecosystem in _parse_manifest(path):
            if not _is_known_ai(ecosystem, name):
                continue
            pkgs.add(_display_name(ecosystem, name))
    return frozenset(pkgs)


class DependencyScanner(BaseScanner):
    name = "dependency_scanner"

    def supports(self, context: ScanContext) -> bool:
        return True

    def scan(
        self, context: ScanContext
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        seen: dict[str, AIComponent] = {}
        for path in _iter_scan_paths(context):
            key = path.name.lower()
            if key not in _MANIFEST_PARSERS and not key.endswith(".csproj"):
                continue
            for name, raw_spec, line_no, pinned_ver, ecosystem in _parse_manifest(path):
                if not _is_known_ai(ecosystem, name):
                    continue
                disp = _display_name(ecosystem, name)
                dedup_key = disp.lower()
                if dedup_key in seen:
                    prev = seen[dedup_key]
                    if not prev.sdk_version and pinned_ver:
                        prev.sdk_version = pinned_ver
                        prev.metadata["version_spec"] = raw_spec
                    continue
                meta: dict[str, Any] = {
                    "ecosystem": ecosystem,
                    "version_spec": raw_spec,
                    "manifest": path.name,
                    "known_ai_package": True,
                }
                comp = AIComponent(
                    name=disp,
                    component_type=AIComponentType.DEPENDENCY,
                    file_path=str(path.resolve()),
                    line_number=line_no,
                    sdk_version=pinned_ver,
                    detection_source=DetectionSource.DEPENDENCY_MANIFEST,
                    metadata=meta,
                )
                seen[dedup_key] = comp
        return list(seen.values()), []
