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

"""Container Scanner -- analyzes container images for AI assets.

Tiered strategy (no hard dependency on external binaries):

* Tier 1: ``syft`` binary on PATH or ``~/.aibom/bin/syft``
* Tier 2: Docker daemon (``docker save`` + Python tarfile parsing)
* Tier 3: Local OCI tarball/layout (pure Python tarfile + JSON parsing)

Does NOT start or run containers. Reads image layers directly.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from ..models import AIComponent, AIComponentType, ComponentRelationship
from ..models.enums import DetectionSource
from ..models.scan import ScanContext
from .base import BaseScanner

_LOGGER = logging.getLogger(__name__)

_AI_PACKAGES: frozenset[str] = frozenset({
    "torch", "tensorflow", "transformers", "openai", "anthropic",
    "langchain", "langchain-core", "langchain-openai", "langchain-community",
    "llama-index", "crewai", "autogen", "dspy", "deepagents",
    "vllm", "triton", "onnxruntime", "tensorrt",
    "huggingface-hub", "datasets", "tokenizers", "accelerate", "peft", "trl",
    "mlflow", "wandb", "bentoml", "litellm",
    "keras", "jax", "flax", "optax",
    "scikit-learn", "xgboost", "lightgbm", "catboost",
    "fastapi", "uvicorn", "gradio", "streamlit",
})

_AI_IMAGES_RE_PARTS = (
    "huggingface", "vllm", "tritonserver", "tensorflow/serving",
    "pytorch/pytorch", "nvcr.io/nvidia", "ollama", "bentoml",
    "sagemaker", "mlflow",
)


class ContainerScanner(BaseScanner):
    name = "container_scanner"

    def supports(self, context: ScanContext) -> bool:
        return bool(context.config.get("container_images"))

    def scan(
        self, context: ScanContext,
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        images = context.config.get("container_images", [])
        if not images:
            return [], []

        components: list[AIComponent] = []
        tier = _detect_tier()
        _LOGGER.info("Container scanner using Tier %d (%s)", tier.level, tier.name)

        for image_ref in images:
            try:
                comps = _scan_image(image_ref, tier)
                components.extend(comps)
            except Exception:
                _LOGGER.warning("Container scan failed for %s", image_ref, exc_info=True)

        return components, []


class _Tier:
    __slots__ = ("level", "name", "syft_path", "has_docker")

    def __init__(self, level: int, name: str, syft_path: str | None = None, has_docker: bool = False):
        self.level = level
        self.name = name
        self.syft_path = syft_path
        self.has_docker = has_docker


def _detect_tier() -> _Tier:
    syft = shutil.which("syft")
    if not syft:
        aibom_syft = Path.home() / ".aibom" / "bin" / "syft"
        if aibom_syft.is_file():
            syft = str(aibom_syft)
    if syft:
        return _Tier(1, "syft", syft_path=syft)

    docker = shutil.which("docker")
    if docker:
        try:
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=5, check=True,
            )
            return _Tier(2, "docker_save", has_docker=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            pass

    return _Tier(3, "tarball_only")


def _scan_image(image_ref: str, tier: _Tier) -> list[AIComponent]:
    if tier.level == 1:
        return _scan_with_syft(image_ref, tier.syft_path or "syft")
    if tier.level == 2:
        return _scan_with_docker_save(image_ref)
    if Path(image_ref).exists():
        return _scan_tarball(image_ref, Path(image_ref))
    _LOGGER.warning("Tier 3 requires a local tarball path, got: %s", image_ref)
    return []


def _scan_with_syft(image_ref: str, syft_path: str) -> list[AIComponent]:
    components: list[AIComponent] = []
    try:
        result = subprocess.run(
            [syft_path, image_ref, "-o", "syft-json", "-q"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            _LOGGER.warning("Syft failed for %s: %s", image_ref, result.stderr[:500])
            return components

        data = json.loads(result.stdout)
        components.extend(_extract_from_syft_json(image_ref, data))
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        _LOGGER.warning("Syft scan error for %s: %s", image_ref, exc)
    return components


def _extract_from_syft_json(image_ref: str, data: dict[str, Any]) -> list[AIComponent]:
    components: list[AIComponent] = []

    source = data.get("source", {})
    config = source.get("metadata", {}).get("config", {})
    env_vars = config.get("Env", [])
    for env_entry in env_vars:
        if isinstance(env_entry, str) and "=" in env_entry:
            k, _, v = env_entry.partition("=")
            if any(kw in k.upper() for kw in ("MODEL", "API_KEY", "OPENAI", "HF_TOKEN")):
                comp_type = AIComponentType.SECRET if "KEY" in k.upper() or "TOKEN" in k.upper() else AIComponentType.MODEL
                components.append(
                    AIComponent(
                        name=k,
                        component_type=comp_type,
                        file_path=f"container:{image_ref}",
                        line_number=0,
                        model_name=v if comp_type == AIComponentType.MODEL else None,
                        framework="docker",
                        detection_source=DetectionSource.CONFIG_FILE,
                        metadata={"container_image": image_ref, "env_var": k, "env_value": v},
                    )
                )

    for artifact in data.get("artifacts", []):
        pkg_name = artifact.get("name", "")
        if pkg_name.lower().replace("_", "-") in _AI_PACKAGES:
            version = artifact.get("version", "")
            components.append(
                AIComponent(
                    name=pkg_name,
                    component_type=AIComponentType.DEPENDENCY,
                    file_path=f"container:{image_ref}",
                    line_number=0,
                    sdk_version=version,
                    framework="pip",
                    detection_source=DetectionSource.DEPENDENCY_MANIFEST,
                    metadata={
                        "container_image": image_ref,
                        "package_type": artifact.get("type", ""),
                    },
                )
            )

    return components


def _scan_with_docker_save(image_ref: str) -> list[AIComponent]:
    components: list[AIComponent] = []
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=True) as tmp:
        try:
            subprocess.run(
                ["docker", "save", "-o", tmp.name, image_ref],
                capture_output=True, timeout=120, check=True,
            )
            return _scan_tarball(image_ref, Path(tmp.name))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            _LOGGER.warning("docker save failed for %s: %s", image_ref, exc)
    return components


def _scan_tarball(image_ref: str, tar_path: Path) -> list[AIComponent]:
    components: list[AIComponent] = []
    try:
        with tarfile.open(tar_path, "r") as tf:
            for member in tf.getmembers():
                if member.name == "manifest.json":
                    f = tf.extractfile(member)
                    if f:
                        manifest = json.loads(f.read())
                        if isinstance(manifest, list) and manifest:
                            config_file = manifest[0].get("Config", "")
                            if config_file:
                                cf = tf.extractfile(config_file)
                                if cf:
                                    config = json.loads(cf.read())
                                    env_vars = config.get("config", {}).get("Env", [])
                                    for env_entry in env_vars:
                                        if isinstance(env_entry, str) and "=" in env_entry:
                                            k, _, v = env_entry.partition("=")
                                            if any(kw in k.upper() for kw in ("MODEL", "API_KEY", "OPENAI", "HF_TOKEN")):
                                                comp_type = AIComponentType.SECRET if "KEY" in k.upper() or "TOKEN" in k.upper() else AIComponentType.MODEL
                                                components.append(
                                                    AIComponent(
                                                        name=k,
                                                        component_type=comp_type,
                                                        file_path=f"container:{image_ref}",
                                                        line_number=0,
                                                        model_name=v if comp_type == AIComponentType.MODEL else None,
                                                        framework="docker",
                                                        detection_source=DetectionSource.CONFIG_FILE,
                                                        metadata={"container_image": image_ref, "env_var": k},
                                                    )
                                                )

                if member.name.endswith("/METADATA") and "site-packages" in member.name:
                    f = tf.extractfile(member)
                    if f:
                        _check_package_metadata(image_ref, member.name, f.read().decode("utf-8", errors="replace"), components)
    except Exception:
        _LOGGER.debug("Tarball scan error for %s", tar_path, exc_info=True)

    return components


def _check_package_metadata(
    image_ref: str, meta_path: str, content: str, components: list[AIComponent],
) -> None:
    name = ""
    version = ""
    for line in content.splitlines():
        if line.startswith("Name:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("Version:"):
            version = line.split(":", 1)[1].strip()
        if name and version:
            break
    if name and name.lower().replace("_", "-") in _AI_PACKAGES:
        components.append(
            AIComponent(
                name=name,
                component_type=AIComponentType.DEPENDENCY,
                file_path=f"container:{image_ref}",
                line_number=0,
                sdk_version=version,
                framework="pip",
                detection_source=DetectionSource.DEPENDENCY_MANIFEST,
                metadata={"container_image": image_ref, "installed_path": meta_path},
            )
        )
