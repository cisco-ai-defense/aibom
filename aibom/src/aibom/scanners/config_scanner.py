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
import logging
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

import yaml  # type: ignore[import-untyped]
from pathspec import PathSpec

from ..models import AIComponent, ComponentRelationship, ScanContext
from ..models.enums import AIComponentType, DetectionSource, RelationshipType
from .base import BaseScanner

_LOGGER = logging.getLogger(__name__)

from ..utils.path_filter import should_skip_dir

_AI_COMPOSE_IMAGE_MARKERS = (
    "ollama/ollama",
    "ollama/",
    "vllm/vllm",
    "vllm-openai",
    "text-generation-inference",
    "tritonserver",
    "localai/localai",
    "localai/",
    "ggerganov/llama.cpp",
    "llama.cpp",
)

_ENV_LINE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s#]*))\s*(?:#.*)?$"
)

_MODEL_NAME_RE = re.compile(
    r"(?i)^(gpt-|claude-|o[1-9]-|gemini-|mistral|meta-llama|llama[-_]?[0-9]|text-embedding|"
    r"command-|jamba-|cohere|azure/|anthropic\.|amazon\.|models/|"
    r"(?:(?:meta-llama|mistralai|google|openai|microsoft|nvidia|huggingface|"
    r"sentence-transformers|bigscience|stabilityai|deepseek|qwen|tiiuae|"
    r"databricks|together|cohere|anthropic)/[a-zA-Z0-9_.-]+))"
)

_NOT_MODEL_RE = re.compile(
    r"(?i)"
    r"(\.com/|\.io/|\.org/|\.dev/|dkr\.ecr|"
    r"\.yaml$|\.yml$|\.json$|\.py$|\.tf$|\.crt$|\.key$|\.pem$|\.sh$|"
    r"^arn:|^https?://|^ssh://|^git@|"
    r"kubernetes\.io/|helm\.sh/|"
    r"^actions/|@v\d+$|"
    r"/secrets?/|/configmaps?/|/templates?/|"
    r"^envoyproxy/|^nginx/|^redis/|^postgres/|^mysql/|^mongo)"
)


def _is_model_name(value: str) -> bool:
    if not value or len(value) > 120:
        return False
    stripped = value.strip()
    if not _MODEL_NAME_RE.match(stripped):
        return False
    if _NOT_MODEL_RE.search(stripped):
        return False
    return True

_FROM_RE = re.compile(r"^\s*FROM\s+(--platform=\S+\s+)?(\S+)", re.IGNORECASE)
_ENV_RE = re.compile(
    r"^\s*ENV\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$",
    re.IGNORECASE,
)
_COPY_MODEL_RE = re.compile(
    r"^\s*COPY\s+.+?\.(gguf|safetensors|onnx|pt|pth|bin|ckpt|h5|pb)\b",
    re.IGNORECASE,
)
_TOML_SECTION = re.compile(
    r"^\s*\[tool\.(langchain|crewai|dspy)([^\]]*)\]\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_DSPY_MODEL_LINE = re.compile(r"(?im)^\s*model\s*=\s*[\"']?([^\"'\s#]+)[\"']?")


def _line_for_substring(text: str, needle: str) -> int:
    idx = text.find(needle)
    if idx < 0:
        return 1
    return text.count("\n", 0, idx) + 1


def _load_exclude_spec(context: ScanContext) -> Optional[PathSpec]:
    if not context.exclude_patterns:
        return None
    return PathSpec.from_lines("gitwildmatch", context.exclude_patterns)


def _is_excluded(abs_file: Path, root: Path, spec: Optional[PathSpec]) -> bool:
    if not spec:
        return False
    try:
        rel = abs_file.relative_to(root).as_posix()
    except ValueError:
        rel = abs_file.as_posix()
    return spec.match_file(rel)


def _is_config_target(name: str) -> bool:
    n = name.lower()
    if n in {
        "langgraph.json",
        "crewai.yaml",
        "crewai.yml",
        "dspy.toml",
        "docker-compose.yml",
        "docker-compose.yaml",
        "dockerfile",
        "pyproject.toml",
    }:
        return True
    if n == ".env" or n.startswith(".env."):
        return True
    if n.endswith(".dockerfile"):
        return True
    return False


def _iter_config_files(context: ScanContext) -> Iterator[tuple[Path, Path]]:
    idx = context.file_index()
    if idx:
        for entries in idx.values():
            for entry in entries:
                if _is_config_target(entry.path.name):
                    yield entry.path, entry.root
        return

    spec = _load_exclude_spec(context)
    for root_str in context.paths:
        root = Path(root_str).resolve()
        if root.is_file():
            if not _is_config_target(root.name):
                continue
            parent = root.parent
            if _is_excluded(root, parent, spec):
                continue
            yield root, parent
            continue
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dp = Path(dirpath)
            rel_root = dp.relative_to(root).as_posix() if dp != root else ""
            dirnames[:] = [
                d
                for d in dirnames
                if not should_skip_dir(d)
                and not any(
                    should_skip_dir(p) for p in (rel_root + "/" + d).split("/") if p
                )
            ]
            for fn in filenames:
                if not _is_config_target(fn):
                    continue
                file_path = dp / fn
                if _is_excluded(file_path, root, spec):
                    continue
                yield file_path, root


def _read_text(path: Path) -> Optional[str]:
    from .file_cache import read_text_cached

    try:
        return read_text_cached(path)
    except OSError as e:
        _LOGGER.debug("Unreadable %s: %s", path, e)
        return None


def _looks_like_model_name(value: str) -> bool:
    v = value.strip().strip("\"'")
    if not v or len(v) > 256:
        return False
    try:
        from .model_detector import _registry_lookup
        if _registry_lookup(v) is not None:
            return True
    except ImportError:
        pass
    if _is_model_name(v):
        return True
    return False


def _provider_from_env_key(key: str) -> Optional[str]:
    ku = key.upper()
    for pref, name in (
        ("OPENAI_", "openai"),
        ("ANTHROPIC_", "anthropic"),
        ("GOOGLE_", "google"),
        ("AZURE_", "azure"),
        ("COHERE_", "cohere"),
        ("MISTRAL_", "mistral"),
    ):
        if ku.startswith(pref):
            return name
    return None


def _docker_image_is_ai(image: str) -> bool:
    img = image.lower().split("@", 1)[0]
    for m in _AI_COMPOSE_IMAGE_MARKERS:
        if m in img:
            return True
    return False


def _parse_langgraph(
    path: Path, text: str
) -> tuple[list[AIComponent], list[ComponentRelationship]]:
    out: list[AIComponent] = []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [], []
    if not isinstance(data, dict):
        return [], []
    graphs = data.get("graphs")
    if isinstance(graphs, dict):
        for gid, spec in graphs.items():
            if not isinstance(gid, str):
                continue
            ln = _line_for_substring(text, f'"{gid}"')
            desc = str(spec) if spec is not None else ""
            out.append(
                AIComponent(
                    name=f"langgraph_graph_{gid}",
                    component_type=AIComponentType.AGENT,
                    file_path=str(path.resolve()),
                    line_number=ln,
                    framework="langgraph",
                    detection_source=DetectionSource.CONFIG_FILE,
                    description=f"LangGraph graph `{gid}` → {desc}",
                    config_source=str(path),
                    metadata={
                        "graph_id": gid,
                        "graph_spec": desc,
                        "config_kind": "langgraph.json",
                    },
                ),
            )
    nodes = data.get("nodes")
    if isinstance(nodes, list):
        for i, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue
            ntype = node.get("type") or node.get("node_type") or "node"
            nid = node.get("id") or node.get("name") or f"node_{i}"
            raw_snip = json.dumps(node, default=str)[:500]
            ln = _line_for_substring(
                text, raw_snip[:40] if len(raw_snip) > 40 else str(nid)
            )
            out.append(
                AIComponent(
                    name=f"langgraph_{nid}",
                    component_type=AIComponentType.AGENT,
                    file_path=str(path.resolve()),
                    line_number=ln,
                    framework="langgraph",
                    detection_source=DetectionSource.CONFIG_FILE,
                    description=f"LangGraph node type `{ntype}`",
                    metadata={"node_type": str(ntype), "config_kind": "langgraph.json"},
                ),
            )
    elif isinstance(nodes, dict):
        for nid, node in nodes.items():
            if not isinstance(node, dict):
                continue
            ntype = node.get("type") or node.get("node_type") or "node"
            ln = _line_for_substring(text, str(nid))
            out.append(
                AIComponent(
                    name=f"langgraph_{nid}",
                    component_type=AIComponentType.AGENT,
                    file_path=str(path.resolve()),
                    line_number=ln,
                    framework="langgraph",
                    detection_source=DetectionSource.CONFIG_FILE,
                    description=f"LangGraph node type `{ntype}`",
                    metadata={"node_type": str(ntype), "config_kind": "langgraph.json"},
                ),
            )

    def _walk_models(obj: Any, depth: int = 0) -> None:
        if depth > 12:
            return
        if isinstance(obj, str) and _looks_like_model_name(obj):
            ln = _line_for_substring(text, obj[: min(40, len(obj))])
            out.append(
                AIComponent(
                    name=obj[:120],
                    component_type=AIComponentType.MODEL,
                    file_path=str(path.resolve()),
                    line_number=ln,
                    framework="langgraph",
                    detection_source=DetectionSource.CONFIG_FILE,
                    model_name=obj.strip().strip("\"'")[:256],
                    metadata={"config_kind": "langgraph.json"},
                ),
            )
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk_models(v, depth + 1)
        elif isinstance(obj, list):
            for v in obj:
                _walk_models(v, depth + 1)

    _walk_models(data)
    return out, []


def _crew_agent_name(ag: dict[str, Any], idx: int) -> str:
    for k in ("name", "role", "id"):
        v = ag.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()[:200]
    return f"agent_{idx}"


def _parse_crewai(
    path: Path, text: str
) -> tuple[list[AIComponent], list[ComponentRelationship]]:
    components: list[AIComponent] = []
    relationships: list[ComponentRelationship] = []
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return [], []
    if not isinstance(data, dict):
        return [], []
    agents_raw = data.get("agents")
    agent_by_name: dict[str, AIComponent] = {}
    if isinstance(agents_raw, list):
        for i, ag in enumerate(agents_raw):
            if not isinstance(ag, dict):
                continue
            nm = _crew_agent_name(ag, i)
            needle = nm if nm in text else (ag.get("role") or nm)
            ln = _line_for_substring(text, str(needle)) if needle else 1
            comp = AIComponent(
                name=f"crewai_agent_{nm}",
                component_type=AIComponentType.AGENT,
                file_path=str(path.resolve()),
                line_number=ln,
                framework="crewai",
                detection_source=DetectionSource.CONFIG_FILE,
                description=f"CrewAI agent `{nm}`",
                config_source=str(path),
                metadata={"config_kind": "crewai", "agent_name": nm},
            )
            components.append(comp)
            agent_by_name[nm] = comp
            tools = ag.get("tools")
            if isinstance(tools, list):
                for t in tools:
                    tname = (
                        str(t)
                        if not isinstance(t, dict)
                        else str(
                            t.get("name") or t.get("tool") or t,
                        )
                    )
                    tln = _line_for_substring(text, tname[:80])
                    tool_comp = AIComponent(
                        name=f"crewai_tool_{tname}",
                        component_type=AIComponentType.TOOL,
                        file_path=str(path.resolve()),
                        line_number=tln,
                        framework="crewai",
                        detection_source=DetectionSource.CONFIG_FILE,
                        description=f"CrewAI tool `{tname}` (agent `{nm}`)",
                        metadata={"tool_name": tname, "agent": nm},
                    )
                    components.append(tool_comp)
                    relationships.append(
                        ComponentRelationship(
                            source_instance_id=comp.instance_id,
                            target_instance_id=tool_comp.instance_id,
                            relationship_type=RelationshipType.USES_TOOL,
                            label=RelationshipType.USES_TOOL.value,
                            source_name=comp.name,
                            target_name=tool_comp.name,
                            source_type=AIComponentType.AGENT,
                            target_type=AIComponentType.TOOL,
                        ),
                    )
    tools_top = data.get("tools")
    if isinstance(tools_top, list):
        for t in tools_top:
            tname = (
                str(t)
                if not isinstance(t, dict)
                else str(
                    t.get("name") or t.get("id") or t,
                )
            )
            tln = _line_for_substring(text, tname[:80])
            components.append(
                AIComponent(
                    name=f"crewai_tool_{tname}",
                    component_type=AIComponentType.TOOL,
                    file_path=str(path.resolve()),
                    line_number=tln,
                    framework="crewai",
                    detection_source=DetectionSource.CONFIG_FILE,
                    description=f"CrewAI tool `{tname}`",
                    metadata={"config_kind": "crewai", "tool_name": tname},
                ),
            )
    tasks = data.get("tasks")
    if isinstance(tasks, list):
        for i, task in enumerate(tasks):
            if not isinstance(task, dict):
                continue
            tid = str(task.get("name") or task.get("id") or f"task_{i}")
            tln = _line_for_substring(text, tid)
            task_comp = AIComponent(
                name=f"crewai_task_{tid}",
                component_type=AIComponentType.AGENT,
                file_path=str(path.resolve()),
                line_number=tln,
                framework="crewai",
                detection_source=DetectionSource.CONFIG_FILE,
                description=f"CrewAI task `{tid}`",
                metadata={"config_kind": "crewai", "task_name": tid},
            )
            components.append(task_comp)
            agent_key = task.get("agent") or task.get("agent_id")
            if isinstance(agent_key, str):
                src = agent_by_name.get(agent_key)
                if src is None:
                    for anm, ac in agent_by_name.items():
                        if agent_key in (anm, f"crewai_agent_{anm}"):
                            src = ac
                            break
                if src is not None:
                    relationships.append(
                        ComponentRelationship(
                            source_instance_id=src.instance_id,
                            target_instance_id=task_comp.instance_id,
                            relationship_type=RelationshipType.CUSTOM,
                            label="TASK_FOR_AGENT",
                            source_name=src.name,
                            target_name=task_comp.name,
                            source_type=AIComponentType.AGENT,
                            target_type=AIComponentType.AGENT,
                        ),
                    )
    return components, relationships


def _parse_dspy_toml(
    path: Path, text: str
) -> tuple[list[AIComponent], list[ComponentRelationship]]:
    out: list[AIComponent] = []
    for m in _DSPY_MODEL_LINE.finditer(text):
        val = m.group(1).strip().strip("\"'")
        if not val:
            continue
        ln = text.count("\n", 0, m.start()) + 1
        out.append(
            AIComponent(
                name=f"dspy_model_{val[:100]}",
                component_type=AIComponentType.MODEL,
                file_path=str(path.resolve()),
                line_number=ln,
                framework="dspy",
                detection_source=DetectionSource.CONFIG_FILE,
                model_name=val[:256],
                description="DSPy model configuration",
                metadata={"config_kind": "dspy.toml"},
            ),
        )
    return out, []


def _parse_env(
    path: Path, text: str
) -> tuple[list[AIComponent], list[ComponentRelationship]]:
    out: list[AIComponent] = []
    for i, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = _ENV_LINE.match(line)
        if not m:
            continue
        key = m.group(1)
        value = next(g for g in m.groups()[1:] if g is not None)
        ku = key.upper()

        if ku.endswith("_API_KEY"):
            out.append(
                AIComponent(
                    name=f"env_secret_{key}",
                    component_type=AIComponentType.SECRET,
                    file_path=str(path.resolve()),
                    line_number=i,
                    detection_source=DetectionSource.CONFIG_FILE,
                    description=f"API key variable `{key}` present (value redacted)",
                    text=None,
                    metadata={
                        "env_var": key,
                        "redacted": True,
                        "config_kind": ".env",
                    },
                ),
            )
            continue

        is_url = value.strip().startswith(("http://", "https://"))
        is_endpoint_key = (
            ku.endswith("_ENDPOINT")
            or ku.endswith("_API_BASE")
            or ku.endswith("_BASE_URL")
        )
        if is_endpoint_key and is_url:
            is_custom_served = any(
                tok in ku for tok in ("SAGEMAKER", "INFERENCE", "SERVING")
            )
            ctype = AIComponentType.MODEL_ENDPOINT if is_custom_served else AIComponentType.LLM_ENDPOINT
            prov = _provider_from_env_key(key)
            out.append(
                AIComponent(
                    name=f"env:{key}",
                    component_type=ctype,
                    file_path=str(path.resolve()),
                    line_number=i,
                    detection_source=DetectionSource.CONFIG_FILE,
                    description=f"Endpoint URL from `{key}` (value redacted)",
                    text=None,
                    framework=prov or "",
                    metadata={
                        "env_var": key,
                        "redacted": True,
                        "config_kind": ".env",
                    },
                ),
            )
            continue

        if is_endpoint_key or ku.endswith("_API_VERSION"):
            prov = _provider_from_env_key(key)
            out.append(
                AIComponent(
                    name=f"env_api_config_{key}",
                    component_type=AIComponentType.MODEL,
                    file_path=str(path.resolve()),
                    line_number=i,
                    detection_source=DetectionSource.CONFIG_FILE,
                    description=f"API configuration `{key}` (value redacted)",
                    text=None,
                    framework=prov or "",
                    metadata={
                        "env_var": key,
                        "redacted": True,
                        "config_kind": ".env",
                    },
                ),
            )
            continue

        if "MODEL" in ku or "LLM" in ku:
            if _looks_like_model_name(value):
                v = value.strip().strip("\"'")
                out.append(
                    AIComponent(
                        name=f"env_model_{key}",
                        component_type=AIComponentType.MODEL,
                        file_path=str(path.resolve()),
                        line_number=i,
                        detection_source=DetectionSource.CONFIG_FILE,
                        model_name=v[:256],
                        description=f"Model from `{key}`",
                        metadata={"env_var": key, "config_kind": ".env"},
                    ),
                )
            continue

        prov = _provider_from_env_key(key)
        if prov:
            out.append(
                AIComponent(
                    name=f"env_provider_{prov}",
                    component_type=AIComponentType.MODEL,
                    file_path=str(path.resolve()),
                    line_number=i,
                    detection_source=DetectionSource.CONFIG_FILE,
                    framework=prov,
                    description=f"Provider env prefix `{key.split('_', 1)[0]}`",
                    text=None,
                    metadata={
                        "env_var": key,
                        "provider": prov,
                        "redacted": True,
                        "config_kind": ".env",
                    },
                ),
            )
    return out, []


def _parse_docker_compose(
    path: Path, text: str
) -> tuple[list[AIComponent], list[ComponentRelationship]]:
    out: list[AIComponent] = []
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return [], []
    if not isinstance(data, dict):
        return [], []
    services = data.get("services")
    if not isinstance(services, dict):
        return [], []
    for svc_name, svc in services.items():
        if not isinstance(svc, dict):
            continue
        image = svc.get("image")
        if not isinstance(image, str) or not _docker_image_is_ai(image):
            continue
        needle = image.split("\n", 1)[0][:80]
        ln = _line_for_substring(text, needle)
        out.append(
            AIComponent(
                name=f"compose_service_{svc_name}",
                component_type=AIComponentType.MODEL,
                file_path=str(path.resolve()),
                line_number=ln,
                detection_source=DetectionSource.CONFIG_FILE,
                model_name=image[:256],
                description=f"Docker Compose service `{svc_name}` uses AI image",
                metadata={
                    "service": str(svc_name),
                    "image": image,
                    "config_kind": "docker-compose",
                },
            ),
        )
    return out, []


def _parse_dockerfile(
    path: Path, text: str
) -> tuple[list[AIComponent], list[ComponentRelationship]]:
    out: list[AIComponent] = []
    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        fm = _FROM_RE.match(line)
        if fm:
            img = fm.group(2)
            if _docker_image_is_ai(img):
                out.append(
                    AIComponent(
                        name=f"dockerfile_from_{i}",
                        component_type=AIComponentType.MODEL,
                        file_path=str(path.resolve()),
                        line_number=i,
                        detection_source=DetectionSource.CONFIG_FILE,
                        model_name=img[:256],
                        description="AI-related base image",
                        metadata={"image": img, "config_kind": "Dockerfile"},
                    ),
                )
        em = _ENV_RE.match(line)
        if em:
            ek = em.group(1)
            ev = em.group(2).strip().strip("\"'")
            ku = ek.upper()
            if "MODEL" in ku or "LLM" in ku:
                if _looks_like_model_name(ev):
                    out.append(
                        AIComponent(
                            name=f"dockerfile_env_{ek}",
                            component_type=AIComponentType.MODEL,
                            file_path=str(path.resolve()),
                            line_number=i,
                            detection_source=DetectionSource.CONFIG_FILE,
                            model_name=ev[:256],
                            metadata={"config_kind": "Dockerfile", "env": ek},
                        ),
                    )
        if _COPY_MODEL_RE.match(line):
            out.append(
                AIComponent(
                    name=f"dockerfile_copy_models_{i}",
                    component_type=AIComponentType.MODEL_ARTIFACT,
                    file_path=str(path.resolve()),
                    line_number=i,
                    detection_source=DetectionSource.CONFIG_FILE,
                    description="COPY of model-like artifact",
                    text=line.strip()[:500],
                    metadata={"config_kind": "Dockerfile"},
                ),
            )
    return out, []


def _parse_pyproject_config(
    path: Path, text: str
) -> tuple[list[AIComponent], list[ComponentRelationship]]:
    out: list[AIComponent] = []
    for m in _TOML_SECTION.finditer(text):
        tool = m.group(1).lower()
        extra = (m.group(2) or "").strip()
        start = m.start()
        ln = text.count("\n", 0, start) + 1
        section_header = f"[tool.{tool}{extra}]"
        next_hdr = re.search(
            r"^\s*\[",
            text[m.end() :],
            re.MULTILINE,
        )
        body_end = m.end() + (next_hdr.start() if next_hdr else len(text) - m.end())
        body = text[m.end() : body_end]

        if tool == "langchain":
            ctype = AIComponentType.AGENT
            framework = "langchain"
        elif tool == "crewai":
            ctype = AIComponentType.AGENT
            framework = "crewai"
        else:
            ctype = AIComponentType.MODEL
            framework = "dspy"

        out.append(
            AIComponent(
                name=f"pyproject_tool_{tool}",
                component_type=ctype,
                file_path=str(path.resolve()),
                line_number=ln,
                framework=framework,
                detection_source=DetectionSource.CONFIG_FILE,
                description=f"pyproject `[tool.{tool}{extra}]` section",
                config_source=str(path),
                metadata={"section": section_header, "config_kind": "pyproject.toml"},
            ),
        )

        for mm in re.finditer(
            r"(?im)^\s*(default_model|model|llm|language_model)\s*=\s*[\"']([^\"']+)[\"']",
            body,
        ):
            mname = mm.group(2).strip()
            if not mname:
                continue
            mln = ln + body.count("\n", 0, mm.start())
            out.append(
                AIComponent(
                    name=f"pyproject_{tool}_model",
                    component_type=AIComponentType.MODEL,
                    file_path=str(path.resolve()),
                    line_number=mln,
                    framework=framework,
                    detection_source=DetectionSource.CONFIG_FILE,
                    model_name=mname[:256],
                    metadata={
                        "section": section_header,
                        "config_kind": "pyproject.toml",
                    },
                ),
            )
    return out, []


def _dispatch(
    path: Path, text: str
) -> tuple[list[AIComponent], list[ComponentRelationship]]:
    n = path.name.lower()
    if n == "langgraph.json":
        return _parse_langgraph(path, text)
    if n in ("crewai.yaml", "crewai.yml"):
        return _parse_crewai(path, text)
    if n == "dspy.toml":
        return _parse_dspy_toml(path, text)
    if n == ".env" or n.startswith(".env."):
        return _parse_env(path, text)
    if n in ("docker-compose.yml", "docker-compose.yaml"):
        return _parse_docker_compose(path, text)
    if n == "dockerfile" or n.endswith(".dockerfile"):
        return _parse_dockerfile(path, text)
    if n == "pyproject.toml":
        return _parse_pyproject_config(path, text)
    return [], []


class ConfigScanner(BaseScanner):
    name = "config_scanner"

    def supports(self, context: ScanContext) -> bool:
        return True

    def scan(
        self, context: ScanContext
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        components: list[AIComponent] = []
        relationships: list[ComponentRelationship] = []
        seen: set[Path] = set()
        for file_path, _root in _iter_config_files(context):
            resolved = file_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            text = _read_text(resolved)
            if text is None:
                continue
            try:
                c, r = _dispatch(resolved, text)
            except Exception:
                _LOGGER.debug("Config parse failed for %s", resolved, exc_info=True)
                continue
            components.extend(c)
            relationships.extend(r)
        return components, relationships
