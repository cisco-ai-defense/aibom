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

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import urlparse

from pathspec import PathSpec

from ..models import AIComponent, ComponentRelationship, ScanContext
from ..models.enums import AIComponentType, DetectionSource
from .base import BaseScanner
from .file_cache import read_text_cached
from .import_context import has_any_ai_imports, has_data_imports, has_ml_imports


@dataclass(frozen=True)
class ConceptPattern:
    concept: AIComponentType
    patterns: list[tuple[re.Pattern[str], str]]
    import_patterns: list[tuple[str, str]]
    description: str


def _build_pathspec(patterns: list[str]) -> Optional[PathSpec]:
    if not patterns:
        return None
    return PathSpec.from_lines("gitwildmatch", patterns)


def _iter_files(context: ScanContext) -> Iterator[tuple[Path, str]]:
    idx = context.file_index()
    if idx:
        for entries in idx.values():
            for entry in entries:
                try:
                    rel = entry.path.relative_to(entry.root).as_posix()
                except ValueError:
                    rel = entry.path.as_posix()
                yield entry.path, rel
        return

    spec = _build_pathspec(context.exclude_patterns)
    for scan_root in context.paths:
        root = Path(scan_root)
        if not root.exists():
            continue
        if root.is_file():
            rel = root.name
            if spec and spec.match_file(rel):
                continue
            yield root, rel
            continue
        base = root.resolve()
        for f in root.rglob("*"):
            if not f.is_file():
                continue
            try:
                rel = f.resolve().relative_to(base).as_posix()
            except ValueError:
                rel = f.as_posix()
            if spec and spec.match_file(rel):
                continue
            yield f, rel


def _line_number(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _normalize_hp_value(raw: str) -> Any:
    s = raw.strip().rstrip(",").strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low == "none":
        return None
    try:
        if "." in s or "e" in low:
            return float(s)
        return int(s)
    except ValueError:
        return s


def _extract_uri_from_line(line: str) -> Optional[str]:
    for m in re.finditer(
        r"(?:s3|gs)://[^\s\"'<>]+|https?://[^\s\"'<>]+", line, re.IGNORECASE
    ):
        u = m.group(0).rstrip(").,;]}'\"")
        if urlparse(u).scheme:
            return u
    return None


_CI_PATH_RE = re.compile(
    r"(?i)(?:s3|aws)\s+cp\s+[^\n]*\.(?:safetensors|gguf|pt|pth|onnx|h5|pb|tflite|bin|mlmodel)\b"
)
_CI_GSUTIL_RE = re.compile(
    r"(?i)gsutil\s+cp\s+[^\n]*\.(?:safetensors|gguf|pt|pth|onnx|h5|pb|tflite|bin|mlmodel)\b"
)
_CI_ARTIFACT_RE = re.compile(
    r"(?i)(?:upload-artifact|actions/upload-artifact|artifacts?:|archiveartifacts)\s*[^\n]*"
    r"(?:safetensors|gguf|\.pt\b|\.pth\b|onnx|\.h5\b|\.pb\b|tflite|\.bin\b|mlmodel|\.parquet|\.csv)"
)
_CI_TRAIN_IMAGE_RE = re.compile(
    r"(?i)(?:image:\s*[\"']?|docker://|docker\.io/)"
    r"(?:tensorflow/tensorflow|pytorch/pytorch|nvidia/cuda|huggingface/|"
    r"ghcr\.io/huggingface)[^\s\"']*"
)

_HP_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\blearning_rate\s*=\s*([^,\n#)]+)"), "learning_rate"),
    (re.compile(r"\bbatch_size\s*=\s*([^,\n#)]+)"), "batch_size"),
    (re.compile(r"\bnum_epochs\s*=\s*([^,\n#)]+)"), "num_epochs"),
    (re.compile(r"\bwarmup_steps\s*=\s*([^,\n#)]+)"), "warmup_steps"),
    (re.compile(r"\bweight_decay\s*=\s*([^,\n#)]+)"), "weight_decay"),
    (re.compile(r"\bmax_seq_length\s*=\s*([^,\n#)]+)"), "max_seq_length"),
    (
        re.compile(r"\bgradient_accumulation_steps\s*=\s*([^,\n#)]+)"),
        "gradient_accumulation_steps",
    ),
    (re.compile(r"\bfp16\s*=\s*([^,\n#)]+)"), "fp16"),
    (re.compile(r"\bbf16\s*=\s*([^,\n#)]+)"), "bf16"),
]

_CONCEPT_PATTERNS: list[ConceptPattern] = [
    ConceptPattern(
        AIComponentType.DATASET,
        [
            (re.compile(r"\bload_dataset\s*\("), "huggingface/datasets"),
            (re.compile(r"\bpd\.read_csv\s*\("), "pandas"),
            (re.compile(r"\bpd\.read_parquet\s*\("), "pandas"),
            (re.compile(r"\btf\.data\.Dataset\b"), "tensorflow"),
            (re.compile(r"\btorch\.utils\.data\.DataLoader\b"), "pytorch"),
            (re.compile(r"\bDataLoader\s*\("), "pytorch"),
            (re.compile(r"\bImageFolder\s*\("), "torchvision"),
            (re.compile(r"\bload_from_disk\s*\("), "huggingface/datasets"),
            (
                re.compile(
                    r"(?i)(?:s3|gs)://[^\s\"']+\.(?:csv|parquet|jsonl|tfrecord)\b"
                ),
                "cloud-storage",
            ),
            (
                re.compile(
                    r"(?i)https?://[^\s\"']*\.blob\.core\.windows\.net[^\s\"']*"
                    r"\.(?:csv|parquet|jsonl|tfrecord)\b"
                ),
                "azure-blob",
            ),
        ],
        [
            ("from datasets import", "huggingface/datasets"),
            ("import datasets", "huggingface/datasets"),
        ],
        "Training or analytics data loading",
    ),
    ConceptPattern(
        AIComponentType.TRAINING_RUN,
        [
            (re.compile(r"\bTrainer\s*\("), "huggingface/transformers"),
            (re.compile(r"\.fit\s*\("), "keras/pytorch"),
            (re.compile(r"\.train\s*\("), "pytorch"),
            (re.compile(r"\bmodel\.compile\s*\("), "tensorflow/keras"),
            (re.compile(r"\bAccelerator\s*\("), "huggingface/accelerate"),
            (re.compile(r"\btrainer\.train\s*\("), "huggingface/transformers"),
            (re.compile(r"\btraining_args\s*="), "huggingface/transformers"),
            (re.compile(r"\bTrainingArguments\s*\("), "huggingface/transformers"),
            (re.compile(r"\bAdam\s*\("), "pytorch/tensorflow"),
            (re.compile(r"\bSGD\s*\("), "pytorch/tensorflow"),
            (re.compile(r"\bAdamW\s*\("), "pytorch/tensorflow"),
            (re.compile(r"\blr_scheduler\b"), "pytorch/tensorflow"),
        ],
        [],
        "Model training execution",
    ),
    ConceptPattern(
        AIComponentType.MODEL_ARTIFACT,
        [
            (
                re.compile(
                    r"(?i)(?:s3|gs)://[^\s\"']+\.(?:safetensors|gguf|onnx|h5|pb|tflite|mlmodel|pt|pth)\b"
                ),
                "cloud-storage",
            ),
            (
                re.compile(
                    r"(?i)https?://[^\s\"']*\.blob\.core\.windows\.net[^\s\"']*"
                    r"\.(?:safetensors|gguf|onnx|h5|pb|tflite|mlmodel|pt|pth)\b"
                ),
                "azure-blob",
            ),
            (
                re.compile(
                    r"(?i)(?:model|weights|checkpoint|pretrained|lfs).{0,160}\.bin\b"
                ),
                "model-weights",
            ),
            (re.compile(r"(?<![A-Za-z0-9_])\.safetensors\b"), "safetensors"),
            (re.compile(r"(?<![A-Za-z0-9_])\.gguf\b"), "gguf"),
            (re.compile(r"(?<![A-Za-z0-9_])\.onnx\b"), "onnx"),
            (re.compile(r"(?<![A-Za-z0-9_])\.h5\b"), "keras"),
            (re.compile(r"(?<![A-Za-z0-9_])\.pb\b"), "tensorflow"),
            (re.compile(r"(?<![A-Za-z0-9_])\.tflite\b"), "tensorflow"),
            (re.compile(r"(?<![A-Za-z0-9_])\.pth\b"), "pytorch"),
            (re.compile(r"(?<![A-Za-z0-9_])\.pt\b"), "pytorch"),
            (re.compile(r"(?<![A-Za-z0-9_])\.mlmodel\b"), "coreml"),
            (re.compile(r"\bmodel\.save\s*\("), "keras/tensorflow"),
            (re.compile(r"\btorch\.save\s*\("), "pytorch"),
            (re.compile(r"\bmodel\.save_pretrained\s*\("), "huggingface/transformers"),
            (re.compile(r"\btf\.saved_model\.save\s*\("), "tensorflow"),
            (re.compile(r"\bexport_to_onnx\s*\("), "onnx"),
        ],
        [],
        "Serialized model weights or export",
    ),
    ConceptPattern(
        AIComponentType.EXPERIMENT_TRACKER,
        [
            (re.compile(r"\bwandb\.(init|log)\s*\("), "wandb"),
            (re.compile(r"\bmlflow\.(start_run|log_metric|log_param|autolog)\s*\("), "mlflow"),
            (re.compile(r"\bmlflow\.(log_metric|log_param|set_tag)\s*\("), "mlflow"),
            (re.compile(r"\bneptune\.(init_run|Run)\s*\("), "neptune"),
            (re.compile(r"\bcomet_ml\.Experiment\s*\("), "comet_ml"),
            (re.compile(r"\bSummaryWriter\s*\("), "tensorboard"),
            (re.compile(r"\btorch\.utils\.tensorboard\b"), "tensorboard"),
            (re.compile(r"\bclearml\.(Task|Logger)\b"), "clearml"),
            (re.compile(r"\btrackio\b"), "trackio"),
            (re.compile(r"\bswanlab\.(init|log)\s*\("), "swanlab"),
        ],
        [
            ("import wandb", "wandb"),
            ("import mlflow", "mlflow"),
            ("from mlflow", "mlflow"),
            ("import comet_ml", "comet_ml"),
            ("from comet_ml", "comet_ml"),
            ("import neptune", "neptune"),
            ("from neptune", "neptune"),
        ],
        "Experiment metrics and run tracking",
    ),
    ConceptPattern(
        AIComponentType.MODEL_REGISTRY,
        [
            (re.compile(r"\bmlflow\.register_model\s*\("), "mlflow"),
            (re.compile(r"\bregister_model\s*\("), "ml-registry"),
            (re.compile(r"\bcreate_model_version\s*\("), "model-registry"),
            (re.compile(r"\btransition_model_version_stage\s*\("), "mlflow"),
            (re.compile(r"\bModelRegistry\b"), "model-registry"),
            (re.compile(r"\bupload_model_version\s*\("), "vertex-ai"),
            (re.compile(r"\bModelVersion\s*\("), "model-registry"),
        ],
        [
            ("from azure.ai.ml.entities import Model", "azure-ml"),
            ("from google.cloud import aiplatform", "vertex-ai"),
        ],
        "Registered model versioning and staging",
    ),
    ConceptPattern(
        AIComponentType.DATA_VERSIONING,
        [
            (re.compile(r"\bdvc\.(pull|push|repro|status)\s*\("), "dvc"),
            (re.compile(r"\bdvc\.api\."), "dvc"),
        ],
        [
            ("import dvc", "dvc"),
            ("from dvc", "dvc"),
            ("import dvc.api", "dvc"),
        ],
        "Data and artifact versioning",
    ),
    ConceptPattern(
        AIComponentType.ML_PIPELINE,
        [
            (re.compile(r"@dsl\.pipeline\b"), "kubeflow"),
            (re.compile(r"@kfp\.dsl\.pipeline\b"), "kubeflow"),
            (re.compile(r"@component\b"), "kubeflow/zenml"),
            (re.compile(r"@step\b"), "zenml/metaflow"),
            (re.compile(r"@task\b"), "prefect/airflow"),
            (re.compile(r"\bfrom airflow import DAG\b"), "airflow"),
            (re.compile(r"\bwith\s+DAG\s*\("), "airflow"),
            (re.compile(r"\bfrom airflow\.decorators import\b"), "airflow"),
            (re.compile(r"\bfrom zenml import step\b"), "zenml"),
            (re.compile(r"\bfrom metaflow import\b"), "metaflow"),
            (re.compile(r"\bfrom prefect import flow\b"), "prefect"),
            (re.compile(r"\bPipelineJob\b"), "vertex-ai"),
            (re.compile(r"apiVersion:\s*argoproj\.io"), "argo"),
        ],
        [
            ("from kfp import dsl", "kubeflow"),
            ("from kfp.dsl import", "kubeflow"),
        ],
        "ML workflow orchestration",
    ),
    ConceptPattern(
        AIComponentType.TRAINING_RUN,
        [
            (re.compile(r"""boto3\.client\s*\(\s*["']sagemaker["']\s*\)"""), "aws-sagemaker"),
            (re.compile(r"""\bcreate_training_job\s*\("""), "aws-sagemaker"),
            (re.compile(r"""\bcreate_endpoint\s*\("""), "aws-sagemaker"),
            (re.compile(r"""\bSageMaker(?:Estimator|Processor|Pipeline)\s*\("""), "aws-sagemaker"),
            (re.compile(r"""\bHuggingFace(?:Processor|Model|Estimator)\s*\("""), "aws-sagemaker"),
            (re.compile(r"""\bCustomJob\s*\("""), "gcp-vertex-ai"),
            (re.compile(r"""\baiplatform\.(?:init|CustomJob|AutoMLTabularTrainingJob)\s*\("""), "gcp-vertex-ai"),
            (re.compile(r"""\bcommand_job\s*=\s*command\s*\(""", re.IGNORECASE), "azure-ml"),
            (re.compile(r"""\bml_client\.(?:jobs\.create_or_update|compute\.begin_create_or_update)\s*\("""), "azure-ml"),
        ],
        [
            ("from sagemaker import", "aws-sagemaker"),
            ("import sagemaker", "aws-sagemaker"),
            ("from sagemaker.huggingface import", "aws-sagemaker"),
            ("from google.cloud import aiplatform", "gcp-vertex-ai"),
            ("from azure.ai.ml import command", "azure-ml"),
            ("from azure.ai.ml import MLClient", "azure-ml"),
        ],
        "Cloud ML training job (SageMaker / Vertex AI / Azure ML)",
    ),
]

_YAML_DATA_SECTION_RE = re.compile(r"^\s*data:\s*(?:#.*)?$")

_K8S_DATA_SECTION_RE = re.compile(
    r"^\s*kind:\s*(ConfigMap|Secret|Deployment|StatefulSet|DaemonSet|Service|"
    r"Ingress|Job|CronJob|ServiceAccount|Role|RoleBinding|ClusterRole|"
    r"ClusterRoleBinding|PersistentVolumeClaim|HorizontalPodAutoscaler)\s*$",
    re.MULTILINE,
)


def _emit(
    concept: AIComponentType,
    framework: str,
    description: str,
    file_path: str,
    line_number: int,
    *,
    detection_source: DetectionSource = DetectionSource.CODE_ANALYSIS,
    hyperparameters: Optional[dict[str, Any]] = None,
    storage_uri: Optional[str] = None,
    text: Optional[str] = None,
    confidence: float = 0.85,
    needs_agentic: bool = False,
    agentic_hint: str = "",
) -> AIComponent:
    hp = hyperparameters or {}
    name = concept.value
    if hp:
        name = f"{concept.value}:{next(iter(hp.keys()))}"
    return AIComponent(
        name=name,
        component_type=concept,
        file_path=file_path,
        line_number=line_number,
        framework=framework,
        detection_source=detection_source,
        confidence=confidence,
        needs_agentic=needs_agentic,
        agentic_hint=agentic_hint,
        description=description,
        text=text,
        storage_uri=storage_uri,
        hyperparameters=hp,
    )


_AMBIGUOUS_TRAINING_FW = frozenset({
    "keras/pytorch", "pytorch/tensorflow", "pytorch",
})

_CLOUD_ML_FW = frozenset({
    "aws-sagemaker", "gcp-vertex-ai", "azure-ml",
})

_AMBIGUOUS_DATASET_FW = frozenset({
    "pandas", "cloud-storage", "azure-blob",
})


def _concept_confidence(
    concept: AIComponentType,
    suffix: str,
    file_has_ml: bool,
    file_has_ai: bool,
    file_has_data: bool,
    framework: str,
) -> tuple[float, bool, str]:
    """Return (confidence, needs_agentic, hint) based on import context."""
    if suffix != ".py":
        return (0.75, False, "")

    if concept == AIComponentType.TRAINING_RUN:
        if framework in _CLOUD_ML_FW:
            return (0.9, False, "")
        if file_has_ml:
            return (0.9, False, "")
        return (0.3, True, f".fit()/{framework} without ML imports in file")

    if concept == AIComponentType.DATASET:
        if file_has_ml or file_has_data:
            return (0.85, False, "")
        if framework in _AMBIGUOUS_DATASET_FW:
            return (0.3, True, f"pd.read_csv/{framework} without ML/data imports")
        return (0.7, False, "")

    if concept == AIComponentType.HYPERPARAMETER:
        if file_has_ml:
            return (0.85, False, "")
        return (0.35, True, "hyperparameter kwarg without ML imports")

    if concept in (AIComponentType.MODEL_ARTIFACT, AIComponentType.DATA_VERSIONING):
        return (0.9, False, "")

    if concept in (AIComponentType.EXPERIMENT_TRACKER, AIComponentType.MODEL_REGISTRY):
        return (0.85, False, "")

    if concept == AIComponentType.ML_PIPELINE:
        if file_has_ai or file_has_ml:
            return (0.85, False, "")
        return (0.5, True, "pipeline pattern without AI/ML imports")

    return (0.85, False, "")


_OPENAPI_MARKER_RE = re.compile(
    r'^\s*(?:openapi|swagger)\s*:', re.MULTILINE,
)


def _is_openapi_spec(text: str) -> bool:
    """Return True when the YAML text appears to be an OpenAPI/Swagger spec."""
    return bool(_OPENAPI_MARKER_RE.search(text[:2000]))


class MLLifecycleDetector(BaseScanner):
    name = "ml_lifecycle_detector"

    def supports(self, context: ScanContext) -> bool:
        return True

    def scan(
        self, context: ScanContext
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        components: list[AIComponent] = []
        seen: set[tuple[Any, ...]] = set()

        def add(comp: AIComponent) -> None:
            key = (
                comp.file_path,
                comp.line_number,
                comp.component_type,
                comp.framework,
                tuple(sorted(comp.hyperparameters.items())),
            )
            if key in seen:
                return
            seen.add(key)
            components.append(comp)

        for fpath, rel in _iter_files(context):
            fp_str = str(fpath)
            suffix = fpath.suffix.lower()
            name_lower = fpath.name.lower()

            try:
                text = read_text_cached(fpath)
            except OSError:
                continue

            file_has_ml = has_ml_imports(text) if suffix == ".py" else False
            file_has_ai = has_any_ai_imports(text) if suffix == ".py" else False
            file_has_data = has_data_imports(text) if suffix == ".py" else False

            is_ci = (
                rel.startswith(".github/workflows/")
                and name_lower.endswith((".yml", ".yaml"))
            ) or name_lower == ".gitlab-ci.yml" or name_lower == "jenkinsfile"
            is_yaml = suffix in (".yaml", ".yml")
            is_jenkins = name_lower == "jenkinsfile"

            if name_lower in ("dvc.yaml", "dvc.lock") or name_lower.endswith(".dvc"):
                add(
                    _emit(
                        AIComponentType.DATA_VERSIONING,
                        "dvc",
                        "DVC project metadata",
                        fp_str,
                        1,
                        detection_source=DetectionSource.CONFIG_FILE,
                    )
                )

            if name_lower == "pipeline.yaml" or name_lower == "pipeline.yml":
                add(
                    _emit(
                        AIComponentType.ML_PIPELINE,
                        "pipeline-config",
                        "Pipeline configuration file",
                        fp_str,
                        1,
                        detection_source=DetectionSource.CONFIG_FILE,
                    )
                )

            if is_yaml and (
                "pipelines.kubeflow.org" in text
                or "kubeflow" in rel.lower()
                or "kubeflow" in text.lower()
            ):
                for i, line in enumerate(text.splitlines(), start=1):
                    if "pipelines.kubeflow.org" in line or "kubeflow.org" in line:
                        add(
                            _emit(
                                AIComponentType.ML_PIPELINE,
                                "kubeflow",
                                "Kubeflow pipeline manifest",
                                fp_str,
                                i,
                                detection_source=DetectionSource.CONFIG_FILE,
                                text=line.strip()[:200],
                            )
                        )
                        break

            is_openapi_spec = is_yaml and _is_openapi_spec(text)

            scan_lines = text.splitlines()
            for line_no, line in enumerate(scan_lines, start=1):
                stripped = line.strip()

                if is_yaml and _YAML_DATA_SECTION_RE.match(line):
                    is_k8s_manifest = _K8S_DATA_SECTION_RE.search(text) is not None
                    if not is_k8s_manifest and not is_openapi_spec:
                        uri = _extract_uri_from_line(line)
                        add(
                            _emit(
                                AIComponentType.DATASET,
                                "yaml-config",
                                "YAML data section",
                                fp_str,
                                line_no,
                                detection_source=DetectionSource.CONFIG_FILE,
                                storage_uri=uri,
                                text=stripped[:200],
                            )
                        )

                if suffix == ".py":
                    hp_conf = 0.85 if file_has_ml else 0.35
                    hp_agentic = not file_has_ml
                    for hp_rx, hp_key in _HP_RULES:
                        for m in hp_rx.finditer(line):
                            raw = m.group(1).strip()
                            val = _normalize_hp_value(raw)
                            comp = _emit(
                                AIComponentType.HYPERPARAMETER,
                                "python-kwargs",
                                f"Hyperparameter {hp_key}",
                                fp_str,
                                line_no,
                                hyperparameters={hp_key: val},
                                text=stripped[:200],
                                confidence=hp_conf,
                                needs_agentic=hp_agentic,
                                agentic_hint=f"'{hp_key}' kwarg without ML imports in file" if hp_agentic else "",
                            )
                            key = (
                                fp_str,
                                line_no,
                                AIComponentType.HYPERPARAMETER,
                                comp.framework,
                                (hp_key, val),
                            )
                            if key not in seen:
                                seen.add(key)
                                components.append(comp)

                scan_concepts = suffix == ".py" or is_ci or is_jenkins or is_yaml
                if not scan_concepts:
                    continue

                for cp in _CONCEPT_PATTERNS:
                    for sub, fw in cp.import_patterns:
                        if sub in line:
                            add(
                                _emit(
                                    cp.concept,
                                    fw,
                                    cp.description,
                                    fp_str,
                                    line_no,
                                    text=stripped[:200],
                                )
                            )
                    for rx, fw in cp.patterns:
                        if rx.search(line):
                            uri = None
                            if cp.concept in (
                                AIComponentType.DATASET,
                                AIComponentType.MODEL_ARTIFACT,
                            ):
                                uri = _extract_uri_from_line(line)

                            ctx_conf, ctx_agentic, ctx_hint = _concept_confidence(
                                cp.concept, suffix, file_has_ml, file_has_ai, file_has_data, fw
                            )
                            add(
                                _emit(
                                    cp.concept,
                                    fw,
                                    cp.description,
                                    fp_str,
                                    line_no,
                                    storage_uri=uri,
                                    text=stripped[:200],
                                    confidence=ctx_conf,
                                    needs_agentic=ctx_agentic,
                                    agentic_hint=ctx_hint,
                                )
                            )

                if is_ci or is_jenkins:
                    if _CI_PATH_RE.search(line) or _CI_GSUTIL_RE.search(line):
                        add(
                            _emit(
                                AIComponentType.MODEL_ARTIFACT,
                                "ci-cd",
                                "CI artifact transfer for model files",
                                fp_str,
                                line_no,
                                detection_source=DetectionSource.CONFIG_FILE,
                                text=stripped[:200],
                            )
                        )
                    if _CI_ARTIFACT_RE.search(line):
                        add(
                            _emit(
                                AIComponentType.MODEL_ARTIFACT,
                                "ci-cd",
                                "CI artifact upload referencing ML assets",
                                fp_str,
                                line_no,
                                detection_source=DetectionSource.CONFIG_FILE,
                                text=stripped[:200],
                            )
                        )
                    if _CI_TRAIN_IMAGE_RE.search(line):
                        add(
                            _emit(
                                AIComponentType.TRAINING_RUN,
                                "ci-cd",
                                "ML training container image reference",
                                fp_str,
                                line_no,
                                detection_source=DetectionSource.CONFIG_FILE,
                                text=stripped[:200],
                            )
                        )

            if is_yaml and "dag" in text.lower() and (
                "airflow" in text.lower() or "schedule" in text.lower()
            ):
                if any("airflow" in ln.lower() for ln in scan_lines[:50]):
                    add(
                        _emit(
                            AIComponentType.ML_PIPELINE,
                            "airflow",
                            "Airflow DAG configuration in YAML",
                            fp_str,
                            1,
                            detection_source=DetectionSource.CONFIG_FILE,
                        )
                    )

        return components, []
