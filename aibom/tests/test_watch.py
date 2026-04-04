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

import time

from aibom.watch import scan_for_changes, seed_watch_state


def test_scan_for_changes_detects_new_files(tmp_path) -> None:
    d = tmp_path / "proj"
    d.mkdir()
    a = d / "a.txt"
    a.write_text("1", encoding="utf-8")
    state = seed_watch_state([str(d)])
    b = d / "b.txt"
    b.write_text("x", encoding="utf-8")
    changed = scan_for_changes([str(d)], state)
    assert str(b.resolve()) in changed


def test_scan_for_changes_detects_modified_files(tmp_path) -> None:
    d = tmp_path / "proj"
    d.mkdir()
    f = d / "f.py"
    f.write_text("a", encoding="utf-8")
    state = seed_watch_state([str(d)])
    time.sleep(0.02)
    f.write_text("b", encoding="utf-8")
    changed = scan_for_changes([str(d)], state)
    assert str(f.resolve()) in changed


def test_scan_for_changes_ignores_unchanged_files(tmp_path) -> None:
    d = tmp_path / "proj"
    d.mkdir()
    f = d / "f.py"
    f.write_text("same", encoding="utf-8")
    state = seed_watch_state([str(d)])
    changed = scan_for_changes([str(d)], state)
    assert changed == set()
