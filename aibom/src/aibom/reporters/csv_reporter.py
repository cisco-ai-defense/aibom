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

import csv
from typing import IO

from ..models import AIComponent, ScanResult
from .base import BaseReporter

_CSV_FIELDS = [
    "name",
    "component_type",
    "file_path",
    "line_number",
    "framework",
    "detection_source",
    "model_name",
    "description",
    "instance_id",
]


def _row(c: AIComponent) -> dict[str, str | int]:
    return {
        "name": c.name,
        "component_type": c.component_type.value,
        "file_path": c.file_path,
        "line_number": c.line_number,
        "framework": c.framework,
        "detection_source": c.detection_source.value,
        "model_name": c.model_name or "",
        "description": c.description or "",
        "instance_id": c.instance_id,
    }


class CsvReporter(BaseReporter):
    name = "csv"
    file_extension = ".csv"

    def render(self, result: ScanResult, output: IO[str]) -> None:
        writer = csv.DictWriter(
            output,
            fieldnames=_CSV_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for c in result.all_components:
            writer.writerow(_row(c))
