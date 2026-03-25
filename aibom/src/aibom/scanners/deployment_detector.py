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
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

import yaml  # type: ignore[import-untyped]
from pathspec import PathSpec

from ..models import AIComponent, ComponentRelationship, ScanContext
from ..models.enums import AIComponentType, DetectionSource
from .base import BaseScanner
from .file_cache import read_text_cached

_LOGGER = logging.getLogger(__name__)

from ..utils.path_filter import should_skip_dir

_AWS_AI_RESOURCES: frozenset[str] = frozenset(
    {
        "AWS::SageMaker::Endpoint",
        "AWS::SageMaker::Model",
        "AWS::SageMaker::NotebookInstance",
        "AWS::SageMaker::EndpointConfig",
        "AWS::Bedrock::Agent",
        "AWS::Bedrock::Guardrail",
        "AWS::Bedrock::KnowledgeBase",
        "AWS::Bedrock::DataSource",
        "aws_sagemaker_endpoint",
        "aws_sagemaker_model",
        "aws_sagemaker_endpoint_configuration",
        "aws_sagemaker_notebook_instance",
        "aws_bedrock_agent_agent",
        "aws_bedrock_guardrail",
        "aws_bedrock_knowledge_base",
        "aws_bedrockagent_data_source",
    }
)

_AZURE_AI_RESOURCES: frozenset[str] = frozenset(
    {
        "Microsoft.CognitiveServices/accounts",
        "Microsoft.MachineLearningServices/workspaces",
        "Microsoft.MachineLearningServices/workspaces/onlineEndpoints",
        "Microsoft.MachineLearningServices/workspaces/computes",
        "azurerm_cognitive_account",
        "azurerm_machine_learning_workspace",
        "azurerm_machine_learning_online_endpoint",
        "azurerm_machine_learning_compute_instance",
    }
)

_GCP_AI_RESOURCES: frozenset[str] = frozenset(
    {
        "google_vertex_ai_endpoint",
        "google_vertex_ai_featurestore",
        "google_ml_engine_model",
        "google_notebooks_instance",
    }
)

_ALL_TF_AI_RESOURCES = _AWS_AI_RESOURCES | _AZURE_AI_RESOURCES | _GCP_AI_RESOURCES

_GPU_INSTANCE_RE = re.compile(
    r"(?i)(ml\.(p[345]d?|g[456]|trn1|inf[12])\.\w+|"
    r"Standard_N[CDV]\w+|"
    r"a2-(ultra|mega|high)gpu-\w+|n1-\w+-gpu\w*|"
    r"g2-standard-\w+)"
)

_AI_CONTAINER_IMAGES: tuple[str, ...] = (
    "ollama/ollama",
    "vllm/vllm-openai",
    "vllm/vllm",
    "ghcr.io/huggingface/text-generation-inference",
    "nvcr.io/nvidia/tritonserver",
    "nvcr.io/nvidia/tensorrt",
    "localai/localai",
    "bentoml/",
    "sagemaker-",
    "huggingface-pytorch-",
    "huggingface-tensorflow-",
)

_MODEL_NAME_RE = re.compile(
    r"(?i)^(gpt-|claude-|o[1-9]-|gemini-|mistral|meta-llama|llama[-_]?[0-9]|"
    r"text-embedding|command-|jamba-|cohere|anthropic\.|amazon\.|"
    r"models/|"
    r"(?:(?:meta-llama|mistralai|google|openai|microsoft|nvidia|huggingface|"
    r"sentence-transformers|bigscience|stabilityai|deepseek|qwen|tiiuae|"
    r"databricks|together|cohere|anthropic)/[a-zA-Z0-9_.-]+))"
)

_NOT_MODEL_RE = re.compile(
    r"(?i)"
    r"(\.com/|\.io/|\.org/|\.dev/|dkr\.ecr|"  # Docker registry domains
    r"\.yaml$|\.yml$|\.json$|\.py$|\.tf$|\.crt$|\.key$|\.pem$|\.sh$|"  # file extensions
    r"^arn:|^https?://|^ssh://|^git@|"  # URIs and ARNs
    r"kubernetes\.io/|helm\.sh/|"  # K8s labels
    r"^actions/|@v\d+$|"  # GitHub Actions
    r"/secrets?/|/configmaps?/|/templates?/|"  # K8s resource paths
    r"^envoyproxy/|^nginx/|^redis/|^postgres/|^mysql/|^mongo)"  # known non-AI images
)


def _is_known_model(value: str) -> bool:
    """Check if a string is a known model ID using the model registry (LiteLLM + HF + builtin)."""
    if not value or len(value) > 256:
        return False
    stripped = value.strip()
    if _NOT_MODEL_RE.search(stripped):
        return False
    from .model_detector import _registry_lookup
    return _registry_lookup(stripped) is not None

_AI_ENV_NAMES: frozenset[str] = frozenset(
    {
        "MODEL_NAME",
        "MODEL_ID",
        "MODEL_PATH",
        "MODEL_ENDPOINT",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HUGGINGFACE_TOKEN",
        "HF_TOKEN",
        "COHERE_API_KEY",
        "MISTRAL_API_KEY",
        "AWS_SAGEMAKER_ENDPOINT",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "GOOGLE_AI_API_KEY",
        "EMBEDDING_MODEL",
        "LLM_MODEL",
        "INFERENCE_ENDPOINT",
    }
)

_SECRET_ENV_NAMES: frozenset[str] = frozenset(
    {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HUGGINGFACE_TOKEN",
        "HF_TOKEN",
        "COHERE_API_KEY",
        "MISTRAL_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "GOOGLE_AI_API_KEY",
    }
)

_TRAINING_IMAGE_HINTS: tuple[str, ...] = (
    "huggingface",
    "sagemaker",
    "pytorch",
    "tensorflow",
    "training",
)

_K8S_WORKLOAD_KINDS = frozenset({"Deployment", "StatefulSet", "Job", "CronJob"})

_TF_RESOURCE_RE = re.compile(r'resource\s+"([^"]+)"\s+"([^"]+)"', re.MULTILINE)
_TF_VARIABLE_RE = re.compile(
    r'variable\s+"([^"]+)"\s*\{[^}]*default\s*=\s*"([^"]*)"', re.DOTALL
)
_TF_STRING_ASSIGN_RE = re.compile(
    r"(?m)^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*\"([^\"]*)\""
)
_TF_LOCALS_RE = re.compile(r"(?m)^\s*locals\s*\{")

_BICEP_RESOURCE_RE = re.compile(r"resource\s+\w+\s+'([^']+)'", re.MULTILINE)
_BICEP_PARAM_RE = re.compile(r"param\s+(\w+)\s+\w+\s*=\s*'([^']*)'", re.MULTILINE)


_MODEL_KEY_HINTS = frozenset({
    "model", "model_name", "model_id", "llm_model", "llm",
    "deployment", "engine", "embedding", "embedding_model",
    "default_model", "chat_model", "completion_model",
})


def _helm_key_suggests_model(key_path: str) -> bool:
    parts = key_path.lower().replace("-", "_").split(".")
    return any(p in _MODEL_KEY_HINTS for p in parts)


_AZURE_DEPLOY_KEY_RE = re.compile(r"(?i)azure", re.IGNORECASE)
_AZURE_ENGINE_TAIL_RE = re.compile(r"(?i)\.(engine|deployment|deployment_name)$")


def _azure_deployment_hint(key_path: str, value: str) -> str:
    """Return a richer agentic hint when the key path looks like an Azure deployment name."""
    if _AZURE_DEPLOY_KEY_RE.search(key_path) and _AZURE_ENGINE_TAIL_RE.search(key_path):
        return (
            f"'{value}' is an Azure OpenAI deployment name (key '{key_path}'); "
            f"the actual model behind this deployment cannot be determined "
            f"without Azure API access or manual confirmation"
        )
    return f"Value '{value}' under key '{key_path}' may be model name"


def _emit(
    name: str,
    comp_type: AIComponentType,
    file_path: str,
    line_number: int,
    *,
    framework: str = "",
    model_name: str | None = None,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
    confidence: float = 1.0,
    needs_agentic: bool = False,
    agentic_hint: str = "",
) -> AIComponent:
    return AIComponent(
        name=name,
        component_type=comp_type,
        file_path=file_path,
        line_number=line_number,
        framework=framework,
        detection_source=DetectionSource.CONFIG_FILE,
        model_name=model_name,
        description=description,
        metadata=metadata or {},
        confidence=confidence,
        needs_agentic=needs_agentic,
        agentic_hint=agentic_hint,
    )


def _line_for_needle(content: str, needle: str) -> int:
    idx = content.find(needle)
    if idx < 0:
        return 1
    return content.count("\n", 0, idx) + 1


def _line_at_offset(content: str, offset: int) -> int:
    if offset < 0 or offset > len(content):
        return 1
    return content.count("\n", 0, offset) + 1


def _build_pathspec(patterns: list[str]) -> Optional[PathSpec]:
    if not patterns:
        return None
    return PathSpec.from_lines("gitwildmatch", patterns)


def _is_excluded(abs_file: Path, root: Path, spec: Optional[PathSpec]) -> bool:
    if not spec:
        return False
    try:
        rel = abs_file.relative_to(root).as_posix()
    except ValueError:
        rel = abs_file.as_posix()
    return spec.match_file(rel)


def _path_has_skip_segment(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    return any(should_skip_dir(p) for p in rel.parts[:-1])


def _iter_target_files(context: ScanContext) -> Iterator[tuple[Path, Path]]:
    idx = context.file_index()
    if idx:
        for entries in idx.values():
            for entry in entries:
                yield entry.path, entry.root
        return

    spec = _build_pathspec(context.exclude_patterns)
    for scan_root in context.paths:
        root = Path(scan_root)
        if not root.exists():
            continue
        base = root if root.is_dir() else root.parent
        if root.is_file():
            if not _is_excluded(root, base, spec):
                yield root, base
            continue
        base = root.resolve()
        for f in root.rglob("*"):
            if not f.is_file():
                continue
            if _path_has_skip_segment(f, root):
                continue
            if _is_excluded(f, root, spec):
                continue
            yield f, root


def _image_matches_ai(image: str) -> bool:
    li = image.lower()
    for m in _AI_CONTAINER_IMAGES:
        ml = m.lower()
        if ml.endswith("/"):
            if ml in li:
                return True
        elif li.startswith(ml) or ml in li:
            return True
    return False


def _is_training_image_hint(image: str) -> bool:
    li = image.lower()
    return any(h in li for h in _TRAINING_IMAGE_HINTS)


def _dedup(components: list[AIComponent]) -> list[AIComponent]:
    seen: set[tuple[str, int, str]] = set()
    out: list[AIComponent] = []
    for c in components:
        key = (c.file_path, c.line_number, c.name)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _balance_braces(text: str, open_idx: int) -> tuple[str, int]:
    depth = 0
    j = open_idx
    while j < len(text):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx : j + 1], j + 1
        j += 1
    return text[open_idx:], len(text)


def _bicep_resource_type(raw: str) -> str:
    if "@" in raw:
        return raw.split("@", 1)[0].strip()
    return raw.strip()


def _is_cloudformation(data: dict[str, Any]) -> bool:
    if "AWSTemplateFormatVersion" in data:
        return True
    res = data.get("Resources")
    if isinstance(res, dict):
        for v in res.values():
            if not isinstance(v, dict):
                continue
            t = v.get("Type", "")
            if isinstance(t, str) and t.startswith("AWS::"):
                return True
    return False


def _is_arm_template(data: dict[str, Any]) -> bool:
    schema = data.get("$schema", "")
    return isinstance(schema, str) and "deploymenttemplate" in schema.lower()


def _is_helm_values_filename(name: str) -> bool:
    n = name.lower()
    if n == "values.yaml" or n == "values.yml":
        return True
    return (n.startswith("values-") and n.endswith((".yaml", ".yml")))


def _is_kubernetes_yaml(content: str, data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    if "apiVersion" in data and "kind" in data:
        return True
    if "apiVersion" in content and "kind:" in content:
        return True
    return False


def _classify_yaml_json(
    path: Path, content: str, data: dict[str, Any]
) -> Optional[str]:
    name = path.name.lower()
    if _is_helm_values_filename(name):
        return "helm"
    if _is_cloudformation(data):
        return "cloudformation"
    if _is_arm_template(data):
        return "arm"
    if _is_kubernetes_yaml(content, data):
        return "k8s"
    if name in ("kustomization.yaml", "kustomization.yml"):
        return "k8s"
    return None


def _neutralize_cfn_yaml_intrinsics(yaml_text: str) -> str:
    t = re.sub(r"!Ref\s+\w+", '"__Ref__"', yaml_text)
    t = re.sub(r"!GetAtt\s+[\w.]+\s+\w+", '"__GetAtt__"', t)
    return t


def _safe_yaml_documents(content: str) -> list[Any]:
    def _load_all(raw: str) -> list[Any]:
        return [d for d in yaml.safe_load_all(raw) if d is not None]

    cfnish = "AWSTemplateFormatVersion" in content or (
        "Resources:" in content and "AWS::" in content
    )
    if cfnish:
        try:
            return _load_all(_neutralize_cfn_yaml_intrinsics(content))
        except yaml.YAMLError as e:
            _LOGGER.debug("YAML parse error (cfn neutralized): %s", e)
    try:
        return _load_all(content)
    except yaml.YAMLError as e:
        _LOGGER.debug("YAML parse error: %s", e)
        return []


def _parse_k8s_yaml(
    file_path: Path, data: dict[str, Any], raw: str = ""
) -> list[AIComponent]:
    fp = str(file_path)
    out: list[AIComponent] = []
    kind = data.get("kind")
    if not isinstance(kind, str):
        return out

    if kind == "Service":
        meta = data.get("metadata") or {}
        ann = meta.get("annotations") or {}
        if isinstance(ann, dict):
            for ak, av in ann.items():
                if isinstance(av, str) and _is_known_model(av):
                    ln = _line_for_needle(raw, av) if raw else 1
                    out.append(
                        _emit(
                            av.strip()[:120],
                            AIComponentType.MODEL,
                            fp,
                            ln,
                            model_name=av.strip(),
                            metadata={"annotation": str(ak)},
                            confidence=0.95,
                        )
                    )
        return out

    if kind in ("ConfigMap", "Secret"):
        blobs: list[dict[str, Any]] = []
        for key in ("data", "stringData"):
            b = data.get(key)
            if isinstance(b, dict):
                blobs.append(b)
        for blob in blobs:
            for ek, ev in blob.items():
                if not isinstance(ek, str):
                    continue
                if ek in _AI_ENV_NAMES and isinstance(ev, str) and ev.strip():
                    ln = _line_for_needle(raw, ev) if raw else 1
                    comp_type = (
                        AIComponentType.SECRET
                        if ek in _SECRET_ENV_NAMES
                        else AIComponentType.MODEL
                    )
                    out.append(
                        _emit(
                            f"env:{ek}",
                            comp_type,
                            fp,
                            ln,
                            model_name=ev.strip() if comp_type == AIComponentType.MODEL else None,
                            metadata={"config_key": ek},
                        )
                    )
                if isinstance(ev, str):
                    stripped = ev.strip()
                    if _is_known_model(stripped):
                        ln = _line_for_needle(raw, ev) if raw else 1
                        out.append(
                            _emit(
                                stripped[:120],
                                AIComponentType.MODEL,
                                fp,
                                ln,
                                model_name=stripped,
                                metadata={"config_key": ek},
                                confidence=0.95,
                            )
                        )
        return out

    if kind in _K8S_WORKLOAD_KINDS:
        spec = data.get("spec") or {}
        template = spec.get("template") or spec
        pod_spec = template.get("spec") or {}
        containers: list[dict[str, Any]] = []
        for c in pod_spec.get("containers") or []:
            if isinstance(c, dict):
                containers.append(c)
        for c in pod_spec.get("initContainers") or []:
            if isinstance(c, dict):
                containers.append(c)

        for container in containers:
            image = container.get("image", "")
            res = container.get("resources") or {}
            limits = res.get("limits") or {}
            gpu_meta: dict[str, Any] = {}
            if isinstance(limits, dict):
                for gk, gv in limits.items():
                    if isinstance(gk, str) and gk in ("nvidia.com/gpu", "amd.com/gpu"):
                        gpu_meta["gpu"] = str(gv)
                        gpu_meta["accelerator"] = (
                            "nvidia" if "nvidia" in gk.lower() else "amd"
                        )
            if isinstance(image, str) and _image_matches_ai(image):
                ln = _line_for_needle(raw, image) if raw else 1
                meta = {"container": container.get("name", ""), "image": image}
                meta.update(gpu_meta)
                out.append(
                    _emit(
                        image[:200],
                        AIComponentType.DEPENDENCY,
                        fp,
                        ln,
                        metadata=meta,
                    )
                )
            for ent in container.get("env") or []:
                if not isinstance(ent, dict):
                    continue
                ename = ent.get("name", "")
                if not isinstance(ename, str) or ename not in _AI_ENV_NAMES:
                    continue
                val = ent.get("value", "")
                if not isinstance(val, str):
                    val = ""
                comp_type = (
                    AIComponentType.SECRET
                    if ename in _SECRET_ENV_NAMES
                    else AIComponentType.MODEL
                )
                ln = _line_for_needle(raw, ename) if raw else 1
                out.append(
                    _emit(
                        f"env:{ename}",
                        comp_type,
                        fp,
                        ln,
                        model_name=val.strip() or None if comp_type == AIComponentType.MODEL else None,
                        metadata={"env": ename},
                    )
                )
            limits_check = res.get("limits") or {}
            gpu = False
            for gk in limits_check:
                if isinstance(gk, str) and gk in ("nvidia.com/gpu", "amd.com/gpu"):
                    gpu = True
                    break
            if isinstance(image, str) and gpu and _is_training_image_hint(image):
                if kind in ("Job", "CronJob"):
                    ln = _line_for_needle(raw, image) if raw else 1
                    out.append(
                        _emit(
                            f"training:{container.get('name', 'job')}",
                            AIComponentType.TRAINING_RUN,
                            fp,
                            ln,
                            metadata={"image": image, "kind": kind},
                        )
                    )

    return out


def _walk_helm_values(
    obj: Any,
    file_path: str,
    out: list[AIComponent],
    key_path: str = "",
) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            lk = k.lower()
            sub = f"{key_path}.{k}" if key_path else k
            if isinstance(v, str):
                stripped = v.strip()
                if _image_matches_ai(v):
                    out.append(
                        _emit(
                            v[:200],
                            AIComponentType.DEPENDENCY,
                            file_path,
                            1,
                            metadata={"helm_key": sub, "image": v},
                        )
                    )
                elif _is_known_model(stripped):
                    out.append(
                        _emit(
                            stripped[:120],
                            AIComponentType.MODEL,
                            file_path,
                            1,
                            model_name=stripped,
                            metadata={"helm_key": sub},
                            confidence=0.95,
                        )
                    )
                elif _helm_key_suggests_model(sub) and stripped and len(stripped) < 120:
                    hint = _azure_deployment_hint(sub, stripped)
                    out.append(
                        _emit(
                            stripped[:120],
                            AIComponentType.MODEL,
                            file_path,
                            1,
                            model_name=stripped,
                            metadata={"helm_key": sub},
                            confidence=0.5,
                            needs_agentic=True,
                            agentic_hint=hint,
                        )
                    )
            elif lk == "gpu" and isinstance(v, int) and v > 0:
                out.append(
                    _emit(
                        f"gpu:{sub}",
                        AIComponentType.TRAINING_RUN,
                        file_path,
                        1,
                        metadata={"gpu_count": v, "helm_key": sub},
                    )
                )
            elif isinstance(v, (dict, list)):
                _walk_helm_values(v, file_path, out, sub)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _walk_helm_values(item, file_path, out, f"{key_path}[{i}]")


def _parse_helm_values(file_path: Path, data: dict[str, Any]) -> list[AIComponent]:
    out: list[AIComponent] = []
    _walk_helm_values(data, str(file_path), out)
    return out


def _parse_terraform_resource_block(
    rtype: str,
    rname: str,
    block: str,
    file_path: str,
    full_content: str,
    block_start: int,
    header_line: int,
) -> list[AIComponent]:
    out: list[AIComponent] = []
    if rtype not in _ALL_TF_AI_RESOURCES:
        return out
    meta = {"terraform_resource_type": rtype, "terraform_resource_name": rname}
    out.append(
        _emit(
            f"{rtype}.{rname}",
            AIComponentType.OTHER,
            file_path,
            header_line,
            description=f"Terraform resource {rtype}",
            metadata=meta,
        )
    )
    for m in _TF_STRING_ASSIGN_RE.finditer(block):
        key, val = m.group(1), m.group(2)
        lk = key.lower()
        abs_off = block_start + m.start()
        ln = _line_at_offset(full_content, abs_off)
        if lk in ("model_id", "model_name", "model_data_url") and val.strip():
            out.append(
                _emit(
                    val[:120],
                    AIComponentType.MODEL,
                    file_path,
                    ln,
                    model_name=val,
                    metadata={**meta, "field": key},
                )
            )
        if lk in ("image", "container_image") and val.strip() and _image_matches_ai(val):
            out.append(
                _emit(
                    val[:200],
                    AIComponentType.DEPENDENCY,
                    file_path,
                    ln,
                    metadata={**meta, "field": key, "image": val},
                )
            )
        if lk == "instance_type" and _GPU_INSTANCE_RE.search(val):
            out.append(
                _emit(
                    f"gpu_instance:{val}",
                    AIComponentType.TRAINING_RUN,
                    file_path,
                    ln,
                    metadata={**meta, "instance_type": val},
                )
            )
    return out


def _parse_terraform_locals(content: str, file_path: str) -> list[AIComponent]:
    out: list[AIComponent] = []
    for lm in _TF_LOCALS_RE.finditer(content):
        brace_idx = content.find("{", lm.start())
        if brace_idx < 0:
            continue
        block, _ = _balance_braces(content, brace_idx)
        for m in _TF_STRING_ASSIGN_RE.finditer(block):
            key = m.group(1)
            val = m.group(2)
            lk = key.lower()
            if not any(
                x in lk for x in ("model", "image", "endpoint", "llm", "embedding", "gpu")
            ):
                continue
            abs_off = brace_idx + m.start()
            ln = _line_at_offset(content, abs_off)
            known = _is_known_model(val)
            if known:
                out.append(
                    _emit(
                        val.strip()[:120],
                        AIComponentType.MODEL,
                        file_path,
                        ln,
                        model_name=val.strip(),
                        metadata={"terraform_local": key},
                        confidence=0.95,
                    )
                )
            if _image_matches_ai(val):
                out.append(
                    _emit(
                        val[:200],
                        AIComponentType.DEPENDENCY,
                        file_path,
                        ln,
                        metadata={"terraform_local": key, "image": val},
                    )
                )
    return out


def _parse_terraform(file_path: Path, content: str) -> list[AIComponent]:
    fp = str(file_path)
    out: list[AIComponent] = []
    for m in _TF_RESOURCE_RE.finditer(content):
        rtype, rname = m.group(1), m.group(2)
        header_line = _line_at_offset(content, m.start())
        brace_idx = content.find("{", m.end())
        if brace_idx < 0:
            continue
        block, _ = _balance_braces(content, brace_idx)
        out.extend(
            _parse_terraform_resource_block(
                rtype, rname, block, fp, content, brace_idx, header_line
            )
        )
    for vm in _TF_VARIABLE_RE.finditer(content):
        var_name, default_val = vm.group(1), vm.group(2)
        known = _is_known_model(default_val)
        if known:
            ln = _line_for_needle(content, vm.group(0))
            out.append(
                _emit(
                    default_val.strip()[:120],
                    AIComponentType.MODEL,
                    fp,
                    ln,
                    model_name=default_val.strip(),
                    metadata={"terraform_variable": var_name},
                    confidence=0.95,
                )
            )
        if _image_matches_ai(default_val):
            ln = _line_for_needle(content, vm.group(0))
            out.append(
                _emit(
                    default_val[:200],
                    AIComponentType.DEPENDENCY,
                    fp,
                    ln,
                    metadata={"terraform_variable": var_name, "image": default_val},
                )
            )
    out.extend(_parse_terraform_locals(content, fp))
    return out


def _walk_cfn_properties(
    props: Any,
    fp: str,
    prefix: str,
    out: list[AIComponent],
) -> None:
    if isinstance(props, dict):
        for pk, pv in props.items():
            if isinstance(pk, str) and pk.lower() in (
                "modelid",
                "model_id",
                "modelname",
                "model_name",
            ):
                if isinstance(pv, str) and pv.strip():
                    out.append(
                        _emit(
                            pv.strip()[:120],
                            AIComponentType.MODEL,
                            fp,
                            1,
                            model_name=pv.strip(),
                            metadata={"cfn_property": f"{prefix}.{pk}"},
                        )
                    )
            if isinstance(pk, str) and pk.lower() in ("instancetype", "instance_type"):
                if isinstance(pv, str) and _GPU_INSTANCE_RE.search(pv):
                    out.append(
                        _emit(
                            f"gpu_instance:{pv}",
                            AIComponentType.TRAINING_RUN,
                            fp,
                            1,
                            metadata={"cfn_property": f"{prefix}.{pk}", "instance_type": pv},
                        )
                    )
            if isinstance(pk, str) and "s3" in pk.lower() and isinstance(pv, str):
                if pv.startswith("s3://") and "model" in pk.lower():
                    out.append(
                        _emit(
                            pv[:200],
                            AIComponentType.MODEL_ARTIFACT,
                            fp,
                            1,
                            storage_uri=pv,
                            metadata={"cfn_property": f"{prefix}.{pk}"},
                        )
                    )
            _walk_cfn_properties(pv, fp, f"{prefix}.{pk}" if prefix else pk, out)
    elif isinstance(props, list):
        for i, item in enumerate(props):
            _walk_cfn_properties(item, fp, f"{prefix}[{i}]", out)


def _parse_cloudformation(file_path: Path, data: dict[str, Any]) -> list[AIComponent]:
    fp = str(file_path)
    out: list[AIComponent] = []
    resources = data.get("Resources")
    if isinstance(resources, dict):
        for rid, rdef in resources.items():
            if not isinstance(rdef, dict):
                continue
            rtype = rdef.get("Type", "")
            if not isinstance(rtype, str) or rtype not in _AWS_AI_RESOURCES:
                continue
            out.append(
                _emit(
                    str(rid),
                    AIComponentType.OTHER,
                    fp,
                    1,
                    description=f"CloudFormation {rtype}",
                    metadata={"cfn_type": rtype, "logical_id": rid},
                )
            )
            props = rdef.get("Properties")
            _walk_cfn_properties(props, fp, str(rid), out)

    params = data.get("Parameters")
    if isinstance(params, dict):
        for pname, pdef in params.items():
            if not isinstance(pdef, dict):
                continue
            default = pdef.get("Default")
            if isinstance(default, str):
                if _is_known_model(default):
                    out.append(
                        _emit(
                            default.strip()[:120],
                            AIComponentType.MODEL,
                            fp,
                            1,
                            model_name=default.strip(),
                            metadata={"cfn_parameter": str(pname)},
                            confidence=0.95,
                        )
                    )
    return out


def _parse_arm_template(file_path: Path, data: dict[str, Any]) -> list[AIComponent]:
    fp = str(file_path)
    out: list[AIComponent] = []
    resources = data.get("resources")
    if isinstance(resources, list):
        for res in resources:
            if not isinstance(res, dict):
                continue
            rtype = res.get("type", "")
            if not isinstance(rtype, str):
                continue
            base_type = rtype.split("@", 1)[0].strip()
            if base_type not in _AZURE_AI_RESOURCES:
                continue
            name = res.get("name", "resource")
            if not isinstance(name, str):
                name = "resource"
            out.append(
                _emit(
                    name,
                    AIComponentType.OTHER,
                    fp,
                    1,
                    description=f"ARM {base_type}",
                    metadata={"arm_type": base_type},
                )
            )
            props = res.get("properties") or {}
            if base_type == "Microsoft.CognitiveServices/accounts" and isinstance(
                props, dict
            ):
                deployments = props.get("deployments")
                if isinstance(deployments, list):
                    for dep in deployments:
                        if not isinstance(dep, dict):
                            continue
                        model = dep.get("model")
                        if isinstance(model, dict):
                            mname = model.get("name") or model.get("format")
                            if isinstance(mname, str) and _is_known_model(mname):
                                out.append(
                                    _emit(
                                        mname.strip()[:120],
                                        AIComponentType.MODEL,
                                        fp,
                                        1,
                                        model_name=mname.strip(),
                                        metadata={"arm_deployment": dep.get("name", "")},
                                        confidence=0.95,
                                    )
                                )
            if isinstance(props, dict):
                sku = props.get("sku")
                if isinstance(sku, dict):
                    sname = sku.get("name", "")
                    if isinstance(sname, str) and _GPU_INSTANCE_RE.search(sname):
                        out.append(
                            _emit(
                                f"sku:{sname}",
                                AIComponentType.TRAINING_RUN,
                                fp,
                                1,
                                metadata={"arm_sku": sname},
                            )
                        )

    parameters = data.get("parameters")
    if isinstance(parameters, dict):
        for pname, pdef in parameters.items():
            if not isinstance(pdef, dict):
                continue
            default = pdef.get("defaultValue") or pdef.get("default")
            if isinstance(default, str) and _is_known_model(default):
                out.append(
                    _emit(
                        default.strip()[:120],
                        AIComponentType.MODEL,
                        fp,
                        1,
                        model_name=default.strip(),
                        metadata={"arm_parameter": str(pname)},
                        confidence=0.95,
                    )
                )
    return out


def _parse_bicep(file_path: Path, content: str) -> list[AIComponent]:
    fp = str(file_path)
    out: list[AIComponent] = []
    for m in _BICEP_RESOURCE_RE.finditer(content):
        raw_type = m.group(1)
        rtype = _bicep_resource_type(raw_type)
        if rtype in _AZURE_AI_RESOURCES:
            ln = _line_for_needle(content, m.group(0))
            out.append(
                _emit(
                    rtype,
                    AIComponentType.OTHER,
                    fp,
                    ln,
                    description="Bicep AI resource",
                    metadata={"bicep_type": rtype},
                )
            )
    for pm in _BICEP_PARAM_RE.finditer(content):
        pname, pval = pm.group(1), pm.group(2)
        if _is_known_model(pval):
            ln = _line_for_needle(content, pm.group(0))
            out.append(
                _emit(
                    pval.strip()[:120],
                    AIComponentType.MODEL,
                    fp,
                    ln,
                    model_name=pval.strip(),
                    metadata={"bicep_param": pname},
                    confidence=0.95,
                )
            )
    return out


def _process_file(path: Path) -> list[AIComponent]:
    fp = str(path)
    suffix = path.suffix.lower()
    name_lower = path.name.lower()

    try:
        content = read_text_cached(path)
    except OSError as e:
        _LOGGER.debug("skip unreadable %s: %s", path, e)
        return []

    if suffix == ".tf":
        try:
            return _parse_terraform(path, content)
        except Exception as e:
            _LOGGER.debug("terraform parse %s: %s", path, e)
            return []

    if suffix == ".bicep":
        try:
            return _parse_bicep(path, content)
        except Exception as e:
            _LOGGER.debug("bicep parse %s: %s", path, e)
            return []

    if suffix == ".json":
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            _LOGGER.debug("json parse %s: %s", path, e)
            return []
        if not isinstance(data, dict):
            return []
        try:
            if _is_cloudformation(data):
                return _parse_cloudformation(path, data)
            if _is_arm_template(data):
                return _parse_arm_template(path, data)
        except Exception as e:
            _LOGGER.debug("json IaC parse %s: %s", path, e)
        return []

    if suffix in (".yaml", ".yml") or name_lower.endswith((".yaml", ".yml")):
        docs = _safe_yaml_documents(content)
        all_out: list[AIComponent] = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            try:
                fmt = _classify_yaml_json(path, content, doc)
                if fmt == "helm":
                    all_out.extend(_parse_helm_values(path, doc))
                elif fmt == "k8s":
                    all_out.extend(_parse_k8s_yaml(path, doc, content))
                elif fmt == "cloudformation":
                    all_out.extend(_parse_cloudformation(path, doc))
                elif fmt == "arm":
                    all_out.extend(_parse_arm_template(path, doc))
            except Exception as e:
                _LOGGER.debug("yaml IaC parse %s: %s", path, e)
        return all_out

    return []


class DeploymentDetector(BaseScanner):
    name = "deployment_detector"

    def supports(self, context: ScanContext) -> bool:
        return True

    def scan(
        self, context: ScanContext
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        components: list[AIComponent] = []
        for fpath, _root in _iter_target_files(context):
            try:
                found = _process_file(fpath)
                components.extend(found)
            except Exception as e:
                _LOGGER.debug("deployment scan %s: %s", fpath, e)
        components = _dedup(components)
        _LOGGER.debug("deployment_detector: %d components", len(components))
        return components, []
