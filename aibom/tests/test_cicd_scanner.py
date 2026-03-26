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

from pathlib import Path

import pytest

from aibom.models.enums import AIComponentType
from aibom.models.scan import ScanContext
from aibom.scanners.cicd_scanner import CICDScanner


@pytest.fixture
def scanner():
    return CICDScanner()


class TestGitHubActions:
    def test_detects_ai_action(self, tmp_path: Path, scanner):
        wf_dir = tmp_path / ".github" / "workflows"
        wf_dir.mkdir(parents=True)
        (wf_dir / "train.yml").write_text(
            "name: train\n"
            "on: push\n"
            "jobs:\n"
            "  train:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: huggingface/push-to-hub@v1\n"
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert any(c.name.startswith("huggingface/") for c in comps)

    def test_detects_ai_step_keywords(self, tmp_path: Path, scanner):
        wf_dir = tmp_path / ".github" / "workflows"
        wf_dir.mkdir(parents=True)
        (wf_dir / "ml.yml").write_text(
            "name: ml\n"
            "on: push\n"
            "jobs:\n"
            "  train:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - name: Train model\n"
            "        run: python train.py --model transformers\n"
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        training_comps = [c for c in comps if c.component_type == AIComponentType.TRAINING_RUN]
        assert len(training_comps) >= 1

    def test_detects_secret_references(self, tmp_path: Path, scanner):
        wf_dir = tmp_path / ".github" / "workflows"
        wf_dir.mkdir(parents=True)
        (wf_dir / "deploy.yml").write_text(
            "name: deploy\n"
            "on: push\n"
            "jobs:\n"
            "  deploy:\n"
            "    runs-on: ubuntu-latest\n"
            "    env:\n"
            "      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}\n"
            "    steps:\n"
            "      - run: echo done\n"
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        secrets = [c for c in comps if c.component_type == AIComponentType.SECRET]
        assert any("OPENAI_API_KEY" in c.name for c in secrets)

    def test_detects_ai_container_image(self, tmp_path: Path, scanner):
        wf_dir = tmp_path / ".github" / "workflows"
        wf_dir.mkdir(parents=True)
        (wf_dir / "serve.yml").write_text(
            "name: serve\n"
            "on: push\n"
            "jobs:\n"
            "  serve:\n"
            "    runs-on: ubuntu-latest\n"
            "    container: vllm/vllm-openai:latest\n"
            "    steps:\n"
            "      - run: echo done\n"
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert any("vllm" in c.name.lower() for c in comps)

    def test_empty_workflow_no_components(self, tmp_path: Path, scanner):
        wf_dir = tmp_path / ".github" / "workflows"
        wf_dir.mkdir(parents=True)
        (wf_dir / "lint.yml").write_text(
            "name: lint\non: push\njobs:\n  lint:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - uses: actions/checkout@v4\n"
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert len(comps) == 0


class TestGitLabCI:
    def test_detects_ai_image(self, tmp_path: Path, scanner):
        (tmp_path / ".gitlab-ci.yml").write_text(
            "train:\n"
            "  image: huggingface/transformers-pytorch-gpu:latest\n"
            "  script:\n"
            "    - python train.py\n"
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert any("huggingface" in c.name.lower() for c in comps)

    def test_detects_training_script(self, tmp_path: Path, scanner):
        (tmp_path / ".gitlab-ci.yml").write_text(
            "finetune:\n"
            "  script:\n"
            "    - python finetune.py --epochs 10\n"
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert any(c.component_type == AIComponentType.TRAINING_RUN for c in comps)
