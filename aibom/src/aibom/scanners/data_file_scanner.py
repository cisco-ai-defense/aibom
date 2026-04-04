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
from pathlib import Path

from ..models import AIComponent, ComponentRelationship, ScanContext
from ..models.enums import AIComponentType, DetectionSource
from .base import BaseScanner

_LOGGER = logging.getLogger(__name__)

_PAR1 = b"PAR1"
_ARROW1 = b"ARROW1"
_MAX_PREFIX = 64

_DATA_EXTENSIONS: frozenset[str] = frozenset({
    ".parquet", ".arrow", ".feather", ".csv", ".tsv",
    ".jsonl", ".ndjson", ".tfrecord",
})


class DataFileScanner(BaseScanner):
    name = "data_file_scanner"

    def supports(self, context: ScanContext) -> bool:
        return True

    def scan(
        self, context: ScanContext,
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        components: list[AIComponent] = []
        idx = context.file_index()

        seen: set[Path] = set()
        for ext in _DATA_EXTENSIONS:
            for entry in idx.get(ext, []):
                p = entry.path.resolve()
                if p in seen:
                    continue
                seen.add(p)
                try:
                    comp = _analyze_data_file(p, ext)
                    if comp:
                        components.append(comp)
                except Exception:
                    _LOGGER.debug("Data file scan failed for %s", p, exc_info=True)

        for scan_root in context.paths:
            root = Path(scan_root)
            if not root.exists() or not root.is_dir():
                continue
            try:
                for dirpath, _dirnames, _filenames in os.walk(root):
                    dp = Path(dirpath)
                    if dp.name.endswith(".lmdb") and dp.is_dir():
                        mdb = dp / "data.mdb"
                        if mdb.is_file():
                            rp = dp.resolve()
                            if rp in seen:
                                continue
                            seen.add(rp)
                            try:
                                st = mdb.stat()
                                components.append(
                                    _dataset_component(
                                        name=dp.name,
                                        file_path=str(dp),
                                        line_number=0,
                                        confidence=0.9,
                                        format_name="lmdb",
                                        size_bytes=st.st_size,
                                    )
                                )
                            except Exception:
                                _LOGGER.debug(
                                    "LMDB scan failed for %s", dp, exc_info=True
                                )
            except Exception:
                _LOGGER.debug("LMDB walk failed under %s", root, exc_info=True)

        return components, []


def _dataset_component(
    *,
    name: str,
    file_path: str,
    line_number: int,
    confidence: float,
    format_name: str,
    size_bytes: int,
) -> AIComponent:
    return AIComponent(
        name=name,
        component_type=AIComponentType.DATASET,
        file_path=file_path,
        line_number=line_number,
        framework="",
        detection_source=DetectionSource.CODE_ANALYSIS,
        confidence=confidence,
        metadata={
            "format": format_name,
            "size_bytes": size_bytes,
        },
    )


def _analyze_data_file(path: Path, ext: str) -> AIComponent | None:
    try:
        st = path.stat()
    except OSError:
        return None
    size_bytes = st.st_size
    name = path.name

    if ext == ".parquet":
        if not _prefix_eq(path, _PAR1):
            return None
        return _dataset_component(
            name=name,
            file_path=str(path),
            line_number=0,
            confidence=0.9,
            format_name="parquet",
            size_bytes=size_bytes,
        )

    if ext in (".arrow", ".feather"):
        if not _prefix_eq(path, _ARROW1):
            return None
        fmt = "arrow" if ext == ".arrow" else "feather"
        return _dataset_component(
            name=name,
            file_path=str(path),
            line_number=0,
            confidence=0.9,
            format_name=fmt,
            size_bytes=size_bytes,
        )

    if ext == ".csv":
        if not _csv_tsv_header_ok(path, comma=True):
            return None
        return _dataset_component(
            name=name,
            file_path=str(path),
            line_number=0,
            confidence=0.9,
            format_name="csv",
            size_bytes=size_bytes,
        )

    if ext == ".tsv":
        if not _csv_tsv_header_ok(path, comma=False):
            return None
        return _dataset_component(
            name=name,
            file_path=str(path),
            line_number=0,
            confidence=0.9,
            format_name="tsv",
            size_bytes=size_bytes,
        )

    if ext in (".jsonl", ".ndjson"):
        if not _jsonl_first_line_object(path):
            return None
        fmt = "jsonl" if ext == ".jsonl" else "ndjson"
        return _dataset_component(
            name=name,
            file_path=str(path),
            line_number=0,
            confidence=0.9,
            format_name=fmt,
            size_bytes=size_bytes,
        )

    if ext == ".tfrecord":
        return _dataset_component(
            name=name,
            file_path=str(path),
            line_number=0,
            confidence=0.7,
            format_name="tfrecord",
            size_bytes=size_bytes,
        )

    return None


def _prefix_eq(path: Path, magic: bytes) -> bool:
    try:
        with path.open("rb") as f:
            buf = f.read(min(_MAX_PREFIX, max(len(magic), 1)))
        return buf[: len(magic)] == magic
    except OSError:
        return False


def _csv_tsv_header_ok(path: Path, *, comma: bool) -> bool:
    try:
        with path.open("rb") as f:
            buf = f.read(8192)
    except OSError:
        return False
    if not buf or b"\x00" in buf[:256]:
        return False
    try:
        text = buf.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = buf.decode("utf-8-sig")
        except UnicodeDecodeError:
            return False
    lines = text.splitlines()
    if not lines:
        return False
    line = lines[0]
    if not line.strip():
        return False
    if comma:
        cols = [c.strip() for c in line.split(",")]
    else:
        cols = [c.strip() for c in line.split("\t")]
    cols = [c for c in cols if c != ""]
    return len(cols) >= 2


def _jsonl_first_line_object(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            buf = f.read(8192)
    except OSError:
        return False
    if not buf:
        return False
    try:
        line = buf.splitlines()[0]
        obj = json.loads(line.decode("utf-8"))
        return isinstance(obj, dict)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, IndexError):
        return False
