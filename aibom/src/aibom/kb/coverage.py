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

"""Shared output projection for uncatalogued AI symbols."""

from __future__ import annotations

import os
from typing import Any

from ..models import ScanResult


def build_coverage_gaps(result: ScanResult) -> dict[str, Any] | None:
    """Group uncatalogued AI symbols by package for reports and requests."""

    grouped: dict[tuple[str, str], set[str]] = {}
    for component in result.all_components:
        metadata = component.metadata
        if not metadata.get("uncatalogued_ai_symbol"):
            continue
        ecosystem = str(metadata.get("ecosystem") or "").strip().lower()
        package_name = str(metadata.get("package_name") or "").strip()
        symbol = str(metadata.get("uncatalogued_symbol") or component.name).strip()
        if ecosystem and package_name and symbol:
            grouped.setdefault((ecosystem, package_name), set()).add(symbol)
    if not grouped:
        return None

    packages = [
        {
            "ecosystem": ecosystem,
            "package_name": package_name,
            "symbols": sorted(symbols),
        }
        for (ecosystem, package_name), symbols in sorted(grouped.items())
    ]
    symbol_count = sum(len(item["symbols"]) for item in packages)
    has_api_key = bool(os.environ.get("CISCO_AI_DEFENSE_API_KEY"))
    return {
        "informational": True,
        "uncatalogued_ai_symbol_count": symbol_count,
        "scan_cache_id": str(result.metadata.get("run_id") or ""),
        "packages": packages,
        "request_hint": (
            "Run `cisco-aibom kb request --from-scan <report.json>` to request "
            "coverage for these symbols."
            if has_api_key
            else "Configure a Cisco AI Defense tenant API key to request KB coverage."
        ),
    }
