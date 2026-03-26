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
import struct
from pathlib import Path

import pytest

from aibom.models.enums import AIComponentType
from aibom.models.scan import ScanContext
from aibom.scanners.model_file_scanner import ModelFileScanner


@pytest.fixture
def scanner():
    return ModelFileScanner()


class TestModelFileDetection:
    def test_detects_safetensors(self, tmp_path: Path, scanner):
        meta = {"__metadata__": {"format": "pt", "model_type": "llama"}}
        header_bytes = json.dumps(meta).encode()
        sf = tmp_path / "model.safetensors"
        with open(sf, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.write(b"\x00" * 100)
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert len(comps) == 1
        assert comps[0].component_type == AIComponentType.MODEL_ARTIFACT
        assert comps[0].metadata["model_format"] == "safetensors"
        assert comps[0].metadata["safetensors_metadata"]["format"] == "pt"

    def test_detects_gguf(self, tmp_path: Path, scanner):
        gguf = tmp_path / "model.gguf"
        with open(gguf, "wb") as f:
            f.write(b"GGUF")
            f.write(struct.pack("<I", 3))
            f.write(b"\x00" * 100)
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert len(comps) == 1
        assert comps[0].metadata["gguf_version"] == 3

    def test_detects_pytorch_by_extension(self, tmp_path: Path, scanner):
        (tmp_path / "checkpoint.pt").write_bytes(b"\x80\x00" * 50)
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert len(comps) == 1
        assert comps[0].metadata["model_format"] == "pytorch"

    def test_skips_non_model_files(self, tmp_path: Path, scanner):
        (tmp_path / "readme.txt").write_text("not a model")
        (tmp_path / "data.csv").write_text("col1,col2\n1,2\n")
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert len(comps) == 0

    def test_invalid_gguf_magic_skipped(self, tmp_path: Path, scanner):
        bad = tmp_path / "bad.gguf"
        bad.write_bytes(b"XXXX" + b"\x00" * 100)
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert len(comps) == 1
        assert "gguf_version" not in comps[0].metadata

    def test_multiple_formats(self, tmp_path: Path, scanner):
        (tmp_path / "a.pt").write_bytes(b"\x00" * 10)
        (tmp_path / "b.h5").write_bytes(b"\x00" * 10)
        (tmp_path / "c.tflite").write_bytes(b"\x00" * 10)
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        formats = {c.metadata["model_format"] for c in comps}
        assert "pytorch" in formats
        assert "tensorflow" in formats
        assert "tflite" in formats
