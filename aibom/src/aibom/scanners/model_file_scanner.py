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

"""Model File Scanner -- detects ML model artifact files on disk.

Identifies common model file formats and extracts available metadata:
* SafeTensors (.safetensors) -- metadata from JSON header
* GGUF (.gguf) -- magic bytes + version from binary header
* ONNX (.onnx) -- magic bytes verification
* PyTorch (.pt, .pth, .bin) -- file presence (no deserialization for safety)
* TensorFlow/Keras (.h5, .pb, .tflite, .keras)
* Core ML (.mlmodel, .mlpackage)

Note: This scanner only inventories model files by format and metadata.
It does NOT deserialize any model files for safety reasons.
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from typing import Any

from ..models import AIComponent, AIComponentType, ComponentRelationship
from ..models.enums import DetectionSource
from ..models.scan import ScanContext
from .base import BaseScanner

_LOGGER = logging.getLogger(__name__)

_MODEL_EXTENSIONS: frozenset[str] = frozenset({
    ".safetensors", ".gguf", ".onnx",
    ".pt", ".pth", ".bin",
    ".h5", ".pb", ".tflite", ".keras",
    ".mlmodel",
})

_EXTENSION_FORMAT: dict[str, str] = {
    ".safetensors": "safetensors",
    ".gguf": "gguf",
    ".onnx": "onnx",
    ".pt": "pytorch",
    ".pth": "pytorch",
    ".bin": "pytorch",
    ".h5": "tensorflow",
    ".pb": "tensorflow",
    ".tflite": "tflite",
    ".keras": "keras",
    ".mlmodel": "coreml",
}


class ModelFileScanner(BaseScanner):
    name = "model_file_scanner"

    def supports(self, context: ScanContext) -> bool:
        return True

    def scan(
        self, context: ScanContext,
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        components: list[AIComponent] = []
        idx = context.file_index()

        model_files: list[Path] = []
        if idx:
            for ext in _MODEL_EXTENSIONS:
                for entry in idx.get(ext, []):
                    model_files.append(entry.path)
        else:
            for scan_path in context.paths:
                root = Path(scan_path)
                if root.is_file():
                    if root.suffix.lower() in _MODEL_EXTENSIONS:
                        model_files.append(root)
                elif root.is_dir():
                    for ext in _MODEL_EXTENSIONS:
                        model_files.extend(root.rglob(f"*{ext}"))

        for mf in model_files:
            try:
                comp = _analyze_model_file(mf)
                if comp:
                    components.append(comp)
            except Exception:
                _LOGGER.debug("Model file scan failed for %s", mf, exc_info=True)

        return components, []


def _analyze_model_file(path: Path) -> AIComponent | None:
    ext = path.suffix.lower()
    fmt = _EXTENSION_FORMAT.get(ext, "unknown")
    file_size = path.stat().st_size

    meta: dict[str, Any] = {
        "model_format": fmt,
        "file_size_bytes": file_size,
        "file_extension": ext,
    }

    if ext == ".safetensors":
        st_meta = _read_safetensors_header(path)
        if st_meta:
            meta["safetensors_metadata"] = st_meta

    elif ext == ".gguf":
        gguf_meta = _read_gguf_header(path)
        if gguf_meta:
            meta.update(gguf_meta)

    elif ext == ".onnx":
        if not _verify_onnx_magic(path):
            return None

    return AIComponent(
        name=path.name,
        component_type=AIComponentType.MODEL_ARTIFACT,
        file_path=str(path),
        line_number=0,
        framework=fmt,
        detection_source=DetectionSource.CODE_ANALYSIS,
        metadata=meta,
    )


def _read_safetensors_header(path: Path) -> dict[str, Any] | None:
    """Read the JSON metadata header from a safetensors file."""
    try:
        with open(path, "rb") as f:
            header_size_bytes = f.read(8)
            if len(header_size_bytes) < 8:
                return None
            header_size = struct.unpack("<Q", header_size_bytes)[0]
            if header_size > 10 * 1024 * 1024:
                return None
            header_bytes = f.read(header_size)
            header = json.loads(header_bytes)
            st_meta = header.get("__metadata__", {})
            if isinstance(st_meta, dict):
                return {k: v for k, v in st_meta.items() if isinstance(v, (str, int, float, bool))}
    except Exception:
        _LOGGER.debug("Failed to read safetensors header from %s", path, exc_info=True)
    return None


def _read_gguf_header(path: Path) -> dict[str, Any] | None:
    """Read the magic and version from a GGUF file header."""
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return None
            version_bytes = f.read(4)
            if len(version_bytes) < 4:
                return {"gguf_version": "unknown"}
            version = struct.unpack("<I", version_bytes)[0]
            return {"gguf_version": version}
    except Exception:
        _LOGGER.debug("Failed to read GGUF header from %s", path, exc_info=True)
    return None


def _verify_onnx_magic(path: Path) -> bool:
    """Check if a file has a valid protobuf header (ONNX uses protobuf)."""
    try:
        with open(path, "rb") as f:
            header = f.read(2)
            return len(header) >= 2 and header[0] == 0x08
    except Exception:
        return False
