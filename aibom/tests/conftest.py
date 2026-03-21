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
from typing import Any

from aibom.models import ScanContext
from aibom.scanners.base import BaseScanner


def run_scanner(
    scanner_class: type[BaseScanner],
    tmp_path: Path,
    files: dict[str, str],
    config: dict[str, Any] | None = None,
) -> tuple[list[Any], list[Any]]:
    """Write *files* into *tmp_path* and run a scanner against it."""
    for rel, content in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    ctx = ScanContext(paths=[str(tmp_path)], config=config or {})
    return scanner_class().scan(ctx)
