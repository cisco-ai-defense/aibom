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

from aibom.benchmark import benchmark_scan, load_ground_truth
from aibom.models import (
    AIComponent,
    AIComponentType,
    ScanResult,
    SourceResult,
)


def _scan(*components: AIComponent) -> ScanResult:
    return ScanResult(
        metadata={},
        sources=[SourceResult(path="/x", components=list(components), relationships=[])],
    )


def test_count_based_from_tmp(tmp_path) -> None:
    p = tmp_path / "gt.yaml"
    p.write_text("components:\n  model:\n    count: 3\n", encoding="utf-8")
    gt = load_ground_truth(p)
    scan = _scan(
        AIComponent(name="a", component_type=AIComponentType.MODEL),
        AIComponent(name="b", component_type=AIComponentType.MODEL),
    )
    r = benchmark_scan(gt, scan, strict_names=False)
    m = r.per_category["model"]
    assert m.gt_count == 3
    assert m.detected_count == 2
    assert m.true_positives == 2
    assert m.false_positives == 0
    assert m.false_negatives == 1
    assert abs(m.precision - 1.0) < 1e-9
    assert abs(m.recall - 2 / 3) < 1e-9


def test_name_based_strict_names_true(tmp_path) -> None:
    p = tmp_path / "gt.yaml"
    p.write_text(
        "components:\n"
        "  tool:\n"
        "    count: 2\n"
        "    names: [Alpha, Beta]\n",
        encoding="utf-8",
    )
    gt = load_ground_truth(p)
    scan = _scan(
        AIComponent(name="alpha", component_type=AIComponentType.TOOL),
        AIComponent(name="gamma", component_type=AIComponentType.TOOL),
    )
    r = benchmark_scan(gt, scan, strict_names=True)
    m = r.per_category["tool"]
    assert m.true_positives == 1
    assert m.false_positives == 1
    assert m.false_negatives == 1


def test_perfect_score(tmp_path) -> None:
    p = tmp_path / "gt.yaml"
    p.write_text("components:\n  agent:\n    count: 2\n", encoding="utf-8")
    gt = load_ground_truth(p)
    scan = _scan(
        AIComponent(name="x", component_type=AIComponentType.AGENT),
        AIComponent(name="y", component_type=AIComponentType.AGENT),
    )
    r = benchmark_scan(gt, scan, strict_names=False)
    m = r.per_category["agent"]
    assert m.true_positives == 2
    assert m.false_positives == 0
    assert m.false_negatives == 0
    assert m.f1 == 1.0
    assert r.overall.f1 == 1.0


def test_zero_detections(tmp_path) -> None:
    p = tmp_path / "gt.yaml"
    p.write_text("components:\n  model:\n    count: 2\n", encoding="utf-8")
    gt = load_ground_truth(p)
    scan = _scan()
    r = benchmark_scan(gt, scan, strict_names=False)
    m = r.per_category["model"]
    assert m.true_positives == 0
    assert m.false_positives == 0
    assert m.false_negatives == 2
    assert m.recall == 0.0


def test_excludes_test_only_metadata(tmp_path) -> None:
    p = tmp_path / "gt.yaml"
    p.write_text("components:\n  model:\n    count: 1\n", encoding="utf-8")
    gt = load_ground_truth(p)
    c = AIComponent(
        name="t",
        component_type=AIComponentType.MODEL,
        metadata={"test_only": True},
    )
    c2 = AIComponent(name="real", component_type=AIComponentType.MODEL)
    scan = _scan(c, c2)
    r = benchmark_scan(gt, scan, strict_names=False)
    assert r.per_category["model"].detected_count == 1


def test_mixed_categories(tmp_path) -> None:
    p = tmp_path / "gt.yaml"
    p.write_text(
        "components:\n"
        "  model:\n    count: 1\n"
        "  tool:\n    count: 2\n",
        encoding="utf-8",
    )
    gt = load_ground_truth(p)
    scan = _scan(
        AIComponent(name="m", component_type=AIComponentType.MODEL),
        AIComponent(name="t1", component_type=AIComponentType.TOOL),
        AIComponent(name="t2", component_type=AIComponentType.TOOL),
        AIComponent(name="t3", component_type=AIComponentType.TOOL),
    )
    r = benchmark_scan(gt, scan, strict_names=False)
    assert r.per_category["model"].true_positives == 1
    assert r.per_category["tool"].true_positives == 2
    assert r.per_category["tool"].false_positives == 1


def test_overall_aggregation(tmp_path) -> None:
    p = tmp_path / "gt.yaml"
    p.write_text(
        "components:\n"
        "  model:\n    count: 2\n"
        "  prompt:\n    count: 1\n",
        encoding="utf-8",
    )
    gt = load_ground_truth(p)
    scan = _scan(
        AIComponent(name="m1", component_type=AIComponentType.MODEL),
        AIComponent(name="p1", component_type=AIComponentType.PROMPT),
    )
    r = benchmark_scan(gt, scan, strict_names=False)
    o = r.overall
    assert o.true_positives == 2
    assert o.false_positives == 0
    assert o.false_negatives == 1
    assert o.gt_count == 3


def test_count_shorthand_yaml(tmp_path) -> None:
    p = tmp_path / "gt.yaml"
    p.write_text("components:\n  dataset: 4\n", encoding="utf-8")
    gt = load_ground_truth(p)
    assert gt.components["dataset"].count == 4
