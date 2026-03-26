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
from unittest.mock import MagicMock, patch

import pytest

from aibom.models.enums import AIComponentType
from aibom.scanners.container_scanner import (
    ContainerScanner,
    _detect_tier,
    _extract_from_syft_json,
)


class TestTierDetection:
    def test_tier3_when_nothing_available(self):
        with patch("shutil.which", return_value=None):
            with patch("pathlib.Path.is_file", return_value=False):
                tier = _detect_tier()
                assert tier.level == 3

    def test_tier1_when_syft_available(self):
        with patch("shutil.which", return_value="/usr/local/bin/syft"):
            tier = _detect_tier()
            assert tier.level == 1
            assert tier.syft_path == "/usr/local/bin/syft"


class TestSyftJsonExtraction:
    def test_extracts_ai_packages(self):
        data = {
            "source": {"metadata": {"config": {"Env": []}}},
            "artifacts": [
                {"name": "torch", "version": "2.5.1", "type": "python"},
                {"name": "numpy", "version": "1.26.0", "type": "python"},
                {"name": "transformers", "version": "4.45.0", "type": "python"},
                {"name": "flask", "version": "3.0.0", "type": "python"},
            ],
        }
        comps = _extract_from_syft_json("test:latest", data)
        names = {c.name for c in comps}
        assert "torch" in names
        assert "transformers" in names
        assert "flask" not in names
        assert "numpy" not in names

    def test_extracts_model_env_vars(self):
        data = {
            "source": {
                "metadata": {
                    "config": {
                        "Env": ["MODEL_NAME=gpt-4", "PATH=/usr/bin"]
                    }
                }
            },
            "artifacts": [],
        }
        comps = _extract_from_syft_json("test:latest", data)
        model_comps = [c for c in comps if c.component_type == AIComponentType.MODEL]
        assert len(model_comps) == 1
        assert model_comps[0].model_name == "gpt-4"

    def test_extracts_secret_env_vars(self):
        data = {
            "source": {
                "metadata": {
                    "config": {
                        "Env": ["OPENAI_API_KEY=sk-abc123"]
                    }
                }
            },
            "artifacts": [],
        }
        comps = _extract_from_syft_json("test:latest", data)
        secret_comps = [c for c in comps if c.component_type == AIComponentType.SECRET]
        assert len(secret_comps) == 1


class TestContainerScannerSupports:
    def test_requires_config(self):
        from aibom.models.scan import ScanContext

        scanner = ContainerScanner()
        ctx = ScanContext(paths=["/tmp/test"])
        assert not scanner.supports(ctx)

        ctx2 = ScanContext(paths=["/tmp/test"], config={"container_images": ["test:latest"]})
        assert scanner.supports(ctx2)
