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
from pathspec import PathSpec

from aibom.models import AIComponent, AIComponentType, ScanContext
from aibom.scanners.base import (
    BaseScanner,
    _load_aibomignore,
    run_scanners,
    scanner_registry,
)

_TEST_SCANNER_CLASSES: list[type[BaseScanner]] = []


def _track_scanner(cls: type[BaseScanner]) -> type[BaseScanner]:
    _TEST_SCANNER_CLASSES.append(cls)
    return cls


@_track_scanner
class RegisteredTestScanner(BaseScanner):
    name = "test_scanner"

    def supports(self, context: ScanContext) -> bool:
        return True

    def scan(self, context: ScanContext):
        return [], []


class UnregisteredEmptyNameScanner(BaseScanner):
    name = ""

    def supports(self, context: ScanContext) -> bool:
        return True

    def scan(self, context: ScanContext):
        return [], []


@pytest.fixture(scope="module", autouse=True)
def _clean_scanner_registry_after_module() -> None:
    yield
    for cls in _TEST_SCANNER_CLASSES:
        while cls in scanner_registry:
            scanner_registry.remove(cls)


@pytest.fixture
def isolated_scanner_registry() -> None:
    saved = list(scanner_registry)
    scanner_registry.clear()
    yield
    scanner_registry.clear()
    scanner_registry.extend(saved)


def test_auto_registration_named_scanner_in_registry():
    assert RegisteredTestScanner in scanner_registry
    assert any(c.name == "test_scanner" for c in scanner_registry)


def test_auto_registration_empty_name_not_registered():
    assert UnregisteredEmptyNameScanner not in scanner_registry


def test_run_scanners_extra_scanner_returns_known_components(isolated_scanner_registry):
    class MockScanner(BaseScanner):
        name = ""

        def supports(self, context: ScanContext) -> bool:
            return True

        def scan(self, context: ScanContext):
            comp = AIComponent(name="c1", component_type=AIComponentType.MODEL)
            return [comp], []

    ctx = ScanContext(paths=["/tmp"])
    comps, rels = run_scanners(ctx, extra_scanners=[MockScanner])
    assert len(comps) == 1
    assert comps[0].name == "c1"
    assert rels == []


def test_run_scanners_multiple_scanners_merge_components(isolated_scanner_registry):
    class ScannerA(BaseScanner):
        name = ""

        def supports(self, context: ScanContext) -> bool:
            return True

        def scan(self, context: ScanContext):
            return [AIComponent(name="a", component_type=AIComponentType.MODEL)], []

    class ScannerB(BaseScanner):
        name = ""

        def supports(self, context: ScanContext) -> bool:
            return True

        def scan(self, context: ScanContext):
            return [AIComponent(name="b", component_type=AIComponentType.AGENT)], []

    ctx = ScanContext(paths=["/tmp"])
    comps, _ = run_scanners(ctx, extra_scanners=[ScannerA, ScannerB])
    names = {c.name for c in comps}
    assert names == {"a", "b"}


def test_run_scanners_failing_scanner_does_not_abort_others(isolated_scanner_registry):
    class BadScanner(BaseScanner):
        name = ""

        def supports(self, context: ScanContext) -> bool:
            return True

        def scan(self, context: ScanContext):
            raise RuntimeError("scanner failed")

    class GoodScanner(BaseScanner):
        name = ""

        def supports(self, context: ScanContext) -> bool:
            return True

        def scan(self, context: ScanContext):
            return [AIComponent(name="ok", component_type=AIComponentType.TOOL)], []

    ctx = ScanContext(paths=["/tmp"])
    comps, _ = run_scanners(ctx, extra_scanners=[BadScanner, GoodScanner])
    assert len(comps) == 1
    assert comps[0].name == "ok"


def test_load_aibomignore_returns_pathspec(tmp_path: Path):
    (tmp_path / ".aibomignore").write_text("*.pyc\nbuild/\n")
    spec = _load_aibomignore([str(tmp_path)])
    assert isinstance(spec, PathSpec)
    assert spec.patterns


def test_load_aibomignore_missing_returns_none(tmp_path: Path):
    assert _load_aibomignore([str(tmp_path)]) is None


def test_supports_filtering_only_supporting_scanner_runs(isolated_scanner_registry):
    class NoSupport(BaseScanner):
        name = ""

        def supports(self, context: ScanContext) -> bool:
            return False

        def scan(self, context: ScanContext):
            raise AssertionError("scan should not run")

    class YesSupport(BaseScanner):
        name = ""

        def supports(self, context: ScanContext) -> bool:
            return True

        def scan(self, context: ScanContext):
            return [AIComponent(name="kept", component_type=AIComponentType.MODEL)], []

    ctx = ScanContext(paths=["/tmp"])
    comps, _ = run_scanners(ctx, extra_scanners=[NoSupport, YesSupport])
    assert len(comps) == 1
    assert comps[0].name == "kept"
