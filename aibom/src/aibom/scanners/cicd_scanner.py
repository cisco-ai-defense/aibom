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

"""CI/CD Scanner -- detects AI-related steps in GitHub Actions and GitLab CI
workflow files.

Looks for:
* AI framework references in step ``run:`` commands and ``uses:`` actions
* Model names in env vars, secrets, and step arguments
* ML pipeline steps (training, evaluation, deployment)
* AI-related Docker images in container/services configuration
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from ..models import AIComponent, AIComponentType, ComponentRelationship
from ..models.enums import DetectionSource
from ..models.scan import ScanContext
from .base import BaseScanner
from .file_cache import read_text_cached

_LOGGER = logging.getLogger(__name__)

_GHA_GLOB = ".github/workflows/*.yml"
_GHA_GLOB2 = ".github/workflows/*.yaml"
_GITLAB_CI = ".gitlab-ci.yml"

_AI_ACTION_PREFIXES = frozenset({
    "huggingface/", "aws-actions/amazon-sagemaker-",
    "google-github-actions/setup-gcloud", "azure/ml-",
    "iterative/setup-dvc", "iterative/cml",
    "bentoml/", "mlflow/",
})

_AI_DOCKER_IMAGES = re.compile(
    r"(?:huggingface|sagemaker|vllm|tritonserver|tensorflow/serving|"
    r"pytorch/pytorch|nvcr\.io/nvidia|ollama|bentoml|mlflow)",
    re.IGNORECASE,
)

_AI_STEP_KEYWORDS = re.compile(
    r"\b(?:train|finetune|fine.tune|evaluate|deploy.model|inference|"
    r"sagemaker|bedrock|vertex.ai|openai|anthropic|huggingface|"
    r"mlflow|wandb|dvc|bentoml|triton|vllm|ollama|"
    r"transformers|torch|tensorflow|keras)\b",
    re.IGNORECASE,
)

_MODEL_REF_RE = re.compile(
    r"\b(?:model|model_name|model_id|MODEL)\s*[:=]\s*['\"]?([a-zA-Z0-9_./-]+)['\"]?",
)

_SECRET_AI_KEYS = re.compile(
    r"\b(?:OPENAI_API_KEY|ANTHROPIC_API_KEY|HUGGINGFACE_TOKEN|HF_TOKEN|"
    r"AWS_ACCESS_KEY|AZURE_OPENAI_KEY|GOOGLE_API_KEY|COHERE_API_KEY|"
    r"WANDB_API_KEY|MLFLOW_TRACKING_TOKEN)\b",
)

_GHA_SECRET_WITH_DEFAULT = re.compile(
    r"\$\{\{\s*secrets\.(\w+)\s*\|\|\s*'([^']+)'\s*\}\}",
)


class CICDScanner(BaseScanner):
    name = "cicd_scanner"

    def supports(self, context: ScanContext) -> bool:
        return True

    def scan(
        self, context: ScanContext,
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        components: list[AIComponent] = []
        idx = context.file_index()

        workflow_files: list[Path] = []
        if idx:
            for ext in (".yml", ".yaml"):
                for entry in idx.get(ext, []):
                    p = entry.path
                    if _is_cicd_file(p):
                        workflow_files.append(p)
        else:
            for scan_path in context.paths:
                root = Path(scan_path)
                gha_dir = root / ".github" / "workflows"
                if gha_dir.is_dir():
                    workflow_files.extend(gha_dir.glob("*.yml"))
                    workflow_files.extend(gha_dir.glob("*.yaml"))
                gitlab = root / ".gitlab-ci.yml"
                if gitlab.is_file():
                    workflow_files.append(gitlab)

        for wf in workflow_files:
            try:
                text = read_text_cached(wf)
            except Exception:
                continue
            try:
                data = yaml.safe_load(text)
            except yaml.YAMLError:
                continue
            if not isinstance(data, dict):
                continue

            if wf.name == ".gitlab-ci.yml":
                components.extend(_scan_gitlab_ci(wf, data, text))
            else:
                components.extend(_scan_github_actions(wf, data, text))

        return components, []


def _is_cicd_file(p: Path) -> bool:
    parts = p.parts
    for i, part in enumerate(parts):
        if part == ".github" and i + 1 < len(parts) and parts[i + 1] == "workflows":
            return True
    return p.name == ".gitlab-ci.yml"


def _scan_github_actions(
    wf: Path, data: dict[str, Any], text: str,
) -> list[AIComponent]:
    components: list[AIComponent] = []
    jobs = data.get("jobs", {})
    if not isinstance(jobs, dict):
        return components

    for _secret in _SECRET_AI_KEYS.finditer(text):
        components.append(
            AIComponent(
                name=_secret.group(),
                component_type=AIComponentType.SECRET,
                file_path=str(wf),
                line_number=_line_of(text, _secret.start()),
                detection_source=DetectionSource.CONFIG_FILE,
                metadata={"cicd_type": "github_actions", "secret_ref": True},
            )
        )

    for job_name, job in jobs.items():
        if not isinstance(job, dict):
            continue

        container = job.get("container")
        if isinstance(container, str) and _AI_DOCKER_IMAGES.search(container):
            components.append(
                AIComponent(
                    name=container,
                    component_type=AIComponentType.DEPENDENCY,
                    file_path=str(wf),
                    line_number=0,
                    framework="docker",
                    detection_source=DetectionSource.CONFIG_FILE,
                    metadata={"cicd_type": "github_actions", "job": job_name, "container_image": True},
                )
            )
        elif isinstance(container, dict):
            image = container.get("image", "")
            if isinstance(image, str) and _AI_DOCKER_IMAGES.search(image):
                components.append(
                    AIComponent(
                        name=image,
                        component_type=AIComponentType.DEPENDENCY,
                        file_path=str(wf),
                        line_number=0,
                        framework="docker",
                        detection_source=DetectionSource.CONFIG_FILE,
                        metadata={"cicd_type": "github_actions", "job": job_name, "container_image": True},
                    )
                )

        for step in job.get("steps", []):
            if not isinstance(step, dict):
                continue
            _scan_gha_step(wf, step, job_name, components)

    return components


def _scan_gha_step(
    wf: Path,
    step: dict[str, Any],
    job_name: str,
    components: list[AIComponent],
) -> None:
    uses = step.get("uses", "")
    if isinstance(uses, str):
        for prefix in _AI_ACTION_PREFIXES:
            if uses.startswith(prefix):
                components.append(
                    AIComponent(
                        name=uses.split("@")[0],
                        component_type=AIComponentType.DEPENDENCY,
                        file_path=str(wf),
                        line_number=0,
                        framework="github_actions",
                        detection_source=DetectionSource.CONFIG_FILE,
                        metadata={"cicd_type": "github_actions", "job": job_name, "action": uses},
                    )
                )
                break

    run_cmd = step.get("run", "")
    if isinstance(run_cmd, str) and _AI_STEP_KEYWORDS.search(run_cmd):
        step_name = step.get("name", "AI-related step")
        is_training = bool(re.search(r"\b(?:train|finetune|fine.tune)\b", run_cmd, re.IGNORECASE))
        comp_type = AIComponentType.TRAINING_RUN if is_training else AIComponentType.ML_PIPELINE
        components.append(
            AIComponent(
                name=step_name,
                component_type=comp_type,
                file_path=str(wf),
                line_number=0,
                detection_source=DetectionSource.CONFIG_FILE,
                metadata={"cicd_type": "github_actions", "job": job_name},
            )
        )

    for m in _MODEL_REF_RE.finditer(str(step)):
        val = m.group(1)
        if len(val) >= 3 and "/" in val or "-" in val:
            components.append(
                AIComponent(
                    name=val,
                    component_type=AIComponentType.MODEL,
                    file_path=str(wf),
                    line_number=0,
                    model_name=val,
                    detection_source=DetectionSource.CONFIG_FILE,
                    heuristic_confidence=0.5,
                    needs_agentic=True,
                    agentic_hint=f"Model reference '{val}' in CI/CD step — verify it is an AI model",
                    metadata={"cicd_type": "github_actions", "job": job_name},
                )
            )

    step_env = step.get("env", {})
    if isinstance(step_env, dict):
        for env_key, env_val in step_env.items():
            if not isinstance(env_val, str):
                continue
            sm = _GHA_SECRET_WITH_DEFAULT.search(env_val)
            if sm:
                secret_name, default_val = sm.group(1), sm.group(2)
                default_val = default_val.strip()
                if default_val and len(default_val) >= 3:
                    from .model_detector import registry_lookup

                    if registry_lookup(default_val) is not None:
                        components.append(
                            AIComponent(
                                name=default_val,
                                component_type=AIComponentType.MODEL,
                                file_path=str(wf),
                                line_number=0,
                                model_name=default_val,
                                detection_source=DetectionSource.CONFIG_FILE,
                                heuristic_confidence=0.7,
                                needs_agentic=True,
                                agentic_hint=f"Default value for ${{{{ secrets.{secret_name} }}}}",
                                metadata={
                                    "cicd_type": "github_actions",
                                    "job": job_name,
                                    "env_var": env_key,
                                    "secret_name": secret_name,
                                },
                            )
                        )


def _scan_gitlab_ci(
    wf: Path, data: dict[str, Any], text: str,
) -> list[AIComponent]:
    components: list[AIComponent] = []

    for _secret in _SECRET_AI_KEYS.finditer(text):
        components.append(
            AIComponent(
                name=_secret.group(),
                component_type=AIComponentType.SECRET,
                file_path=str(wf),
                line_number=_line_of(text, _secret.start()),
                detection_source=DetectionSource.CONFIG_FILE,
                metadata={"cicd_type": "gitlab_ci", "secret_ref": True},
            )
        )

    for job_name, job in data.items():
        if not isinstance(job, dict):
            continue
        if job_name.startswith(".") or job_name in ("stages", "variables", "default", "include", "workflow"):
            continue

        image = job.get("image", "")
        if isinstance(image, str) and _AI_DOCKER_IMAGES.search(image):
            components.append(
                AIComponent(
                    name=image,
                    component_type=AIComponentType.DEPENDENCY,
                    file_path=str(wf),
                    line_number=0,
                    framework="docker",
                    detection_source=DetectionSource.CONFIG_FILE,
                    metadata={"cicd_type": "gitlab_ci", "job": job_name, "container_image": True},
                )
            )
        elif isinstance(image, dict):
            img_name = image.get("name", "")
            if isinstance(img_name, str) and _AI_DOCKER_IMAGES.search(img_name):
                components.append(
                    AIComponent(
                        name=img_name,
                        component_type=AIComponentType.DEPENDENCY,
                        file_path=str(wf),
                        line_number=0,
                        framework="docker",
                        detection_source=DetectionSource.CONFIG_FILE,
                        metadata={"cicd_type": "gitlab_ci", "job": job_name, "container_image": True},
                    )
                )

        for script_key in ("script", "before_script", "after_script"):
            scripts = job.get(script_key, [])
            if isinstance(scripts, list):
                for cmd in scripts:
                    if isinstance(cmd, str) and _AI_STEP_KEYWORDS.search(cmd):
                        is_training = bool(re.search(r"\b(?:train|finetune|fine.tune)\b", cmd, re.IGNORECASE))
                        comp_type = AIComponentType.TRAINING_RUN if is_training else AIComponentType.ML_PIPELINE
                        components.append(
                            AIComponent(
                                name=job_name,
                                component_type=comp_type,
                                file_path=str(wf),
                                line_number=0,
                                detection_source=DetectionSource.CONFIG_FILE,
                                metadata={"cicd_type": "gitlab_ci", "job": job_name},
                            )
                        )
                        break

    return components


def _line_of(text: str, offset: int) -> int:
    return text[:offset].count("\n") + 1
