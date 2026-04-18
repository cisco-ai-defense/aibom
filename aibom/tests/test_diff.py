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
from io import StringIO
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aibom.cli import app
from aibom.diff import DiffResult, diff_scan_results, load_scan_result_json, render_diff_json
from aibom.models import AIComponent, AIComponentType, ScanResult, SourceResult
from aibom.reporters.json_reporter import JsonReporter


def _scan(components: list[AIComponent]) -> ScanResult:
    return ScanResult(
        metadata={},
        sources=[SourceResult(path="/src", components=components, relationships=[])],
        errors=[],
    )


def _write_json(path: Path, result: ScanResult) -> None:
    buf = StringIO()
    JsonReporter().render(result, buf)
    path.write_text(buf.getvalue(), encoding="utf-8")


def test_identical_reports_no_changes() -> None:
    comps = [
        AIComponent(
            name="m1",
            component_type=AIComponentType.MODEL,
            file_path="a.py",
            heuristic_confidence=0.9,
            model_name="gpt-4",
            needs_agentic=False,
            framework="openai",
        ),
    ]
    old = _scan(comps)
    new = _scan([c.model_copy(deep=True) for c in comps])
    d = diff_scan_results(old, new)
    assert d.added == []
    assert d.removed == []
    assert d.changed == []
    assert d.summary["added_count"] == 0
    assert d.summary["removed_count"] == 0
    assert d.summary["changed_count"] == 0
    assert d.summary["unchanged_count"] == 1


def test_added_only() -> None:
    old = _scan([])
    new = _scan([
        AIComponent(name="x", component_type=AIComponentType.AGENT, file_path="b.py"),
    ])
    d = diff_scan_results(old, new)
    assert len(d.added) == 1
    assert d.added[0].name == "x"
    assert d.removed == []
    assert d.changed == []
    assert d.summary["added_count"] == 1
    assert d.summary["unchanged_count"] == 0
    assert d.summary["by_type"]["added"] == {"agent": 1}


def test_removed_only() -> None:
    old = _scan([
        AIComponent(name="gone", component_type=AIComponentType.TOOL, file_path="t.py"),
    ])
    new = _scan([])
    d = diff_scan_results(old, new)
    assert len(d.removed) == 1
    assert d.removed[0].name == "gone"
    assert d.added == []
    assert d.changed == []
    assert d.summary["removed_count"] == 1
    assert d.summary["by_type"]["removed"] == {"tool": 1}


def test_changed_confidence_model_name_needs_agentic() -> None:
    old = _scan([
        AIComponent(
            name="m",
            component_type=AIComponentType.MODEL,
            file_path="f.py",
            heuristic_confidence=0.5,
            model_name="old-model",
            needs_agentic=False,
            framework="x",
        ),
    ])
    new = _scan([
        AIComponent(
            name="m",
            component_type=AIComponentType.MODEL,
            file_path="f.py",
            heuristic_confidence=0.99,
            model_name="new-model",
            needs_agentic=True,
            framework="x",
        ),
    ])
    d = diff_scan_results(old, new)
    assert d.added == []
    assert d.removed == []
    assert len(d.changed) == 1
    ch = d.changed[0]
    assert set(ch.changes) == {"heuristic_confidence", "model_name", "needs_agentic"}
    assert ch.before.model_name == "old-model"
    assert ch.after.model_name == "new-model"


def test_changed_framework_and_file_path() -> None:
    old = _scan([
        AIComponent(
            name="z",
            component_type=AIComponentType.EMBEDDING,
            file_path="old.py",
            framework="faiss",
        ),
    ])
    new = _scan([
        AIComponent(
            name="z",
            component_type=AIComponentType.EMBEDDING,
            file_path="new.py",
            framework="chroma",
        ),
    ])
    d = diff_scan_results(old, new)
    assert len(d.changed) == 1
    assert set(d.changed[0].changes) == {"framework", "file_path"}


def test_mixed_add_remove_change() -> None:
    old = _scan([
        AIComponent(name="keep", component_type=AIComponentType.MODEL, file_path="a.py", heuristic_confidence=1.0),
        AIComponent(name="remove-me", component_type=AIComponentType.PROMPT, file_path="b.py"),
        AIComponent(name="mutate", component_type=AIComponentType.AGENT, file_path="c.py", framework="a"),
    ])
    new = _scan([
        AIComponent(name="keep", component_type=AIComponentType.MODEL, file_path="a.py", heuristic_confidence=1.0),
        AIComponent(name="mutate", component_type=AIComponentType.AGENT, file_path="c.py", framework="b"),
        AIComponent(name="fresh", component_type=AIComponentType.MODEL, file_path="d.py"),
    ])
    d = diff_scan_results(old, new)
    assert {c.name for c in d.added} == {"fresh"}
    assert {c.name for c in d.removed} == {"remove-me"}
    assert len(d.changed) == 1
    assert d.changed[0].before.name == "mutate"
    assert d.summary["added_count"] == 1
    assert d.summary["removed_count"] == 1
    assert d.summary["changed_count"] == 1
    assert d.summary["unchanged_count"] == 1


def test_per_type_summary_accuracy() -> None:
    old = _scan([
        AIComponent(name="m1", component_type=AIComponentType.MODEL, file_path="1.py"),
        AIComponent(name="m2", component_type=AIComponentType.MODEL, file_path="2.py"),
    ])
    new = _scan([
        AIComponent(name="m1", component_type=AIComponentType.MODEL, file_path="1.py"),
        AIComponent(name="m3", component_type=AIComponentType.MODEL, file_path="3.py", heuristic_confidence=0.2),
    ])
    d = diff_scan_results(old, new)
    bt = d.summary["by_type"]
    assert bt["added"] == {"model": 1}
    assert bt["removed"] == {"model": 1}
    assert bt["changed"] == {}
    assert bt["unchanged"] == {"model": 1}


def test_diff_result_json_roundtrip_structure() -> None:
    old = _scan([])
    new = _scan([AIComponent(name="n", component_type=AIComponentType.OTHER, file_path="p.py")])
    d = diff_scan_results(old, new)
    raw = render_diff_json(d)
    data = json.loads(raw)
    assert data["summary"]["added_count"] == 1
    assert len(data["added"]) == 1
    assert data["added"][0]["name"] == "n"


def test_empty_reports() -> None:
    d = diff_scan_results(_scan([]), _scan([]))
    assert d.summary["added_count"] == 0
    assert d.summary["removed_count"] == 0
    assert d.summary["changed_count"] == 0
    assert d.summary["unchanged_count"] == 0
    assert d.summary["by_type"]["added"] == {}
    assert d.summary["by_type"]["unchanged"] == {}


def test_load_scan_result_json_grouped_components(tmp_path: Path) -> None:
    p = tmp_path / "out.json"
    scan = _scan([
        AIComponent(name="c1", component_type=AIComponentType.MODEL, file_path="x.py"),
    ])
    _write_json(p, scan)
    loaded = load_scan_result_json(p)
    assert len(loaded.all_components) == 1
    assert loaded.all_components[0].name == "c1"


def test_cli_diff_json_format(tmp_path: Path) -> None:
    old_p = tmp_path / "old.json"
    new_p = tmp_path / "new.json"
    _write_json(old_p, _scan([]))
    _write_json(
        new_p,
        _scan([AIComponent(name="solo", component_type=AIComponentType.TOOL, file_path="z.py")]),
    )
    runner = CliRunner()
    result = runner.invoke(app, ["diff", "run", str(old_p), str(new_p), "--format", "json"])
    assert result.exit_code == 0
    out = json.loads(result.stdout)
    assert out["summary"]["added_count"] == 1
    assert out["added"][0]["name"] == "solo"


def test_cli_diff_markdown(tmp_path: Path) -> None:
    old_p = tmp_path / "o.json"
    new_p = tmp_path / "n.json"
    _write_json(old_p, _scan([]))
    _write_json(new_p, _scan([AIComponent(name="a", component_type=AIComponentType.MODEL, file_path="f.py")]))
    runner = CliRunner()
    result = runner.invoke(app, ["diff", "run", str(old_p), str(new_p), "-f", "markdown"])
    assert result.exit_code == 0
    assert "## Summary" in result.stdout
    assert "Added" in result.stdout


def test_cli_diff_table_runs(tmp_path: Path) -> None:
    old_p = tmp_path / "o.json"
    new_p = tmp_path / "n.json"
    _write_json(old_p, _scan([]))
    _write_json(new_p, _scan([]))
    runner = CliRunner()
    result = runner.invoke(app, ["diff", "run", str(old_p), str(new_p)])
    assert result.exit_code == 0
    assert "Diff summary" in result.stdout


def test_cli_invalid_format(tmp_path: Path) -> None:
    old_p = tmp_path / "o.json"
    new_p = tmp_path / "n.json"
    _write_json(old_p, _scan([]))
    _write_json(new_p, _scan([]))
    runner = CliRunner()
    result = runner.invoke(app, ["diff", "run", str(old_p), str(new_p), "--format", "xml"])
    assert result.exit_code == 1
