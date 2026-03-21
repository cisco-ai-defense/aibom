# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from aibom.models import AIComponentType
from aibom.scanners.ml_lifecycle_detector import MLLifecycleDetector

from .conftest import run_scanner


class TestMLLifecycleDetector:
    def test_dataset_load_dataset_imdb(self, tmp_path: Path) -> None:
        code = 'from datasets import load_dataset\nds = load_dataset("imdb")\n'
        comps, _ = run_scanner(MLLifecycleDetector, tmp_path, {"train.py": code})
        assert any(c.component_type == AIComponentType.DATASET for c in comps)

    def test_training_run_trainer_train(self, tmp_path: Path) -> None:
        code = "from transformers import Trainer\nTrainer(model=m).train()\n"
        comps, _ = run_scanner(MLLifecycleDetector, tmp_path, {"run.py": code})
        assert any(c.component_type == AIComponentType.TRAINING_RUN for c in comps)

    def test_hyperparameter_learning_rate(self, tmp_path: Path) -> None:
        code = "args = dict(learning_rate=3e-5)\n"
        comps, _ = run_scanner(MLLifecycleDetector, tmp_path, {"hp.py": code})
        hps = [c for c in comps if c.component_type == AIComponentType.HYPERPARAMETER]
        assert any(c.hyperparameters.get("learning_rate") == 3e-5 for c in hps)

    def test_model_artifact_safetensors_reference(self, tmp_path: Path) -> None:
        code = 'weights_uri = "s3://bucket/models/w/.safetensors"\n'
        comps, _ = run_scanner(MLLifecycleDetector, tmp_path, {"paths.py": code})
        assert any(c.component_type == AIComponentType.MODEL_ARTIFACT for c in comps)

    def test_experiment_tracker_wandb(self, tmp_path: Path) -> None:
        code = "import wandb\nwandb.init(project='p')\nwandb.log({'loss': 0.1})\n"
        comps, _ = run_scanner(MLLifecycleDetector, tmp_path, {"track.py": code})
        assert any(c.component_type == AIComponentType.EXPERIMENT_TRACKER for c in comps)

    def test_model_registry_mlflow_register(self, tmp_path: Path) -> None:
        code = "import mlflow\nmlflow.register_model('runs:/x/model', 'ProdModel')\n"
        comps, _ = run_scanner(MLLifecycleDetector, tmp_path, {"reg.py": code})
        assert any(c.component_type == AIComponentType.MODEL_REGISTRY for c in comps)

    def test_data_versioning_dvc_yaml(self, tmp_path: Path) -> None:
        yml = "stages:\n  prepare:\n    cmd: python prep.py\n"
        comps, _ = run_scanner(MLLifecycleDetector, tmp_path, {"dvc.yaml": yml})
        assert any(c.component_type == AIComponentType.DATA_VERSIONING for c in comps)

    def test_ml_pipeline_dsl_decorator(self, tmp_path: Path) -> None:
        code = "from kfp import dsl\n\n@dsl.pipeline(name='p')\ndef p():\n    pass\n"
        comps, _ = run_scanner(MLLifecycleDetector, tmp_path, {"pipe.py": code})
        assert any(c.component_type == AIComponentType.ML_PIPELINE for c in comps)
