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

import os
import time
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.table import Table

from .diff import diff_scan_results
from .models import ScanResult, SourceResult


class WatchState:
    def __init__(self) -> None:
        self.mtimes: dict[str, float] = {}


def _record_mtime(fp: Path, state: WatchState) -> None:
    try:
        state.mtimes[str(fp.resolve())] = fp.stat().st_mtime
    except OSError:
        pass


def seed_watch_state(paths: list[str]) -> WatchState:
    state = WatchState()
    for root in paths:
        p = Path(root).resolve()
        if not p.exists():
            continue
        if p.is_file():
            _record_mtime(p, state)
            continue
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", ".venv", "node_modules"}]
            for fn in filenames:
                _record_mtime(Path(dirpath) / fn, state)
    return state


def scan_for_changes(paths: list[str], state: WatchState) -> set[str]:
    changed: set[str] = set()
    for root in paths:
        p = Path(root).resolve()
        if not p.exists():
            continue
        if p.is_file():
            _check_one(p, state, changed)
            continue
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", ".venv", "node_modules"}]
            for fn in filenames:
                fp = Path(dirpath) / fn
                _check_one(fp, state, changed)
    return changed


def _check_one(fp: Path, state: WatchState, changed: set[str]) -> None:
    key = str(fp.resolve())
    try:
        mtime = fp.stat().st_mtime
    except OSError:
        return
    prev = state.mtimes.get(key)
    state.mtimes[key] = mtime
    if prev is None or mtime != prev:
        changed.add(key)


def _pipeline_result_to_scan_result(
    scan_paths: list[str],
    result: object,
) -> ScanResult:
    components = getattr(result, "components", [])
    relationships = getattr(result, "relationships", [])
    return ScanResult(
        metadata={},
        sources=[
            SourceResult(
                path=scan_paths[0] if len(scan_paths) == 1 else ",".join(scan_paths),
                components=list(components),
                relationships=list(relationships),
            )
        ],
    )


def watch_loop(
    paths: list[str],
    scan_fn: Callable[[], object],
    *,
    console: Console,
    interval: float = 2.0,
    debounce: float = 0.5,
) -> None:
    state = WatchState()
    resolved = [str(Path(p).resolve()) for p in paths]

    first = scan_fn()
    prev: ScanResult | None = _pipeline_result_to_scan_result(resolved, first)
    console.print("[green]Initial scan complete.[/] Watching for changes… (Ctrl+C to stop)")

    while True:
        time.sleep(interval)
        changed = scan_for_changes(resolved, state)
        if not changed:
            continue
        time.sleep(debounce)
        scan_for_changes(resolved, state)
        while True:
            extra = scan_for_changes(resolved, state)
            if not extra:
                break
            time.sleep(debounce)
        result = scan_fn()
        cur = _pipeline_result_to_scan_result(resolved, result)
        if prev is not None:
            diff = diff_scan_results(prev, cur)
            if diff.added or diff.removed:
                console.print("[bold cyan]Change detected — re-scan[/]")
                if diff.added:
                    t = __import__("rich.table", fromlist=["Table"]).Table(title="Added components")
                    t.add_column("Name")
                    t.add_column("Type")
                    for c in diff.added:
                        ct = c.component_type.value
                        t.add_row(c.name, ct)
                    console.print(t)
                if diff.removed:
                    t = __import__("rich.table", fromlist=["Table"]).Table(title="Removed components")
                    t.add_column("Name")
                    t.add_column("Type")
                    for c in diff.removed:
                        ct = c.component_type.value
                        t.add_row(c.name, ct)
                    console.print(t)
            else:
                console.print("[dim]Re-scan: no component delta.[/]")
        prev = cur
