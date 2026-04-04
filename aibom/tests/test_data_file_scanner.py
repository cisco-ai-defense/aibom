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
from aibom.scanners.data_file_scanner import DataFileScanner


@pytest.fixture
def scanner() -> DataFileScanner:
    return DataFileScanner()


class TestDataFileScanner:
    def test_parquet_valid_magic(self, tmp_path: Path, scanner: DataFileScanner) -> None:
        f = tmp_path / "d.parquet"
        f.write_bytes(b"PAR1" + b"\x00" * 20)
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert len(comps) == 1
        assert comps[0].component_type == AIComponentType.DATASET
        assert comps[0].confidence == 0.9
        assert comps[0].metadata["format"] == "parquet"

    def test_parquet_invalid_magic(self, tmp_path: Path, scanner: DataFileScanner) -> None:
        f = tmp_path / "bad.parquet"
        f.write_bytes(b"XXXX" + b"\x00" * 20)
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert len(comps) == 0

    def test_csv_with_header(self, tmp_path: Path, scanner: DataFileScanner) -> None:
        (tmp_path / "data.csv").write_text("col1,col2\n1,2\n", encoding="utf-8")
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert len(comps) == 1
        assert comps[0].component_type == AIComponentType.DATASET
        assert comps[0].metadata["format"] == "csv"

    def test_csv_binary_no_detection(self, tmp_path: Path, scanner: DataFileScanner) -> None:
        f = tmp_path / "bin.csv"
        f.write_bytes(b"\x00\x01\x02\x03" * 20)
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert len(comps) == 0

    def test_jsonl_valid_first_line(self, tmp_path: Path, scanner: DataFileScanner) -> None:
        (tmp_path / "lines.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert len(comps) == 1
        assert comps[0].metadata["format"] == "jsonl"

    def test_arrow_magic(self, tmp_path: Path, scanner: DataFileScanner) -> None:
        f = tmp_path / "t.arrow"
        f.write_bytes(b"ARROW1" + b"\x00" * 10)
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert len(comps) == 1
        assert comps[0].metadata["format"] == "arrow"
        assert comps[0].confidence == 0.9

    def test_feather_magic(self, tmp_path: Path, scanner: DataFileScanner) -> None:
        f = tmp_path / "t.feather"
        f.write_bytes(b"ARROW1" + b"\x00" * 10)
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert len(comps) == 1
        assert comps[0].metadata["format"] == "feather"

    def test_metadata_format_and_size_bytes(
        self, tmp_path: Path, scanner: DataFileScanner,
    ) -> None:
        f = tmp_path / "d.parquet"
        payload = b"PAR1" + b"\x00" * 30
        f.write_bytes(payload)
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert len(comps) == 1
        assert comps[0].metadata["format"] == "parquet"
        assert comps[0].metadata["size_bytes"] == len(payload)

    def test_lmdb_directory(self, tmp_path: Path, scanner: DataFileScanner) -> None:
        lmdb_dir = tmp_path / "store.lmdb"
        lmdb_dir.mkdir()
        (lmdb_dir / "data.mdb").write_bytes(b"x")
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert len(comps) == 1
        assert comps[0].name == "store.lmdb"
        assert comps[0].metadata["format"] == "lmdb"
