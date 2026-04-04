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
import io
import json
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from .models import AIComponentType, ScanResult


class CategoryGT(BaseModel):
    count: int
    names: Optional[list[str]] = None


class GroundTruth(BaseModel):
    components: dict[str, CategoryGT] = Field(default_factory=dict)


class CategoryMetrics(BaseModel):
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    gt_count: int
    detected_count: int


class OverallMetrics(BaseModel):
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    gt_count: int
    detected_count: int


class BenchmarkResult(BaseModel):
    per_category: dict[str, CategoryMetrics]
    overall: OverallMetrics


def load_ground_truth(path: str | Path) -> GroundTruth:
    raw = Path(path).read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Ground truth YAML must be a mapping")
    comps = data.get("components", data)
    if not isinstance(comps, dict):
        raise ValueError("Ground truth must define 'components' as a mapping")
    normalized: dict[str, CategoryGT] = {}
    for k, v in comps.items():
        if isinstance(v, int):
            normalized[k] = CategoryGT(count=v)
        else:
            normalized[k] = CategoryGT.model_validate(v)
    return GroundTruth(components=normalized)


def _prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return precision, recall, f1


def _non_test_components(scan: ScanResult) -> list:
    from .models.scan import AIComponent

    out: list[AIComponent] = []
    for c in scan.all_components:
        if c.metadata.get("test_only"):
            continue
        out.append(c)
    return out


def _by_type(components: list) -> dict[str, list]:
    by: dict[str, list] = {}
    for c in components:
        ct = c.component_type
        key = ct.value if isinstance(ct, AIComponentType) else str(ct)
        by.setdefault(key, []).append(c)
    return by


def benchmark_scan(
    gt: GroundTruth,
    scan: ScanResult,
    strict_names: bool = False,
) -> BenchmarkResult:
    detected = _non_test_components(scan)
    by_type = _by_type(detected)

    per_category: dict[str, CategoryMetrics] = {}
    sum_tp = sum_fp = sum_fn = 0
    sum_gt = sum_det = 0

    for cat, gt_cat in gt.components.items():
        gt_count = gt_cat.count
        if strict_names and gt_cat.names:
            gt_set = {n.lower() for n in gt_cat.names}
            gt_count = len(gt_set)
            det_list = by_type.get(cat, [])
            det_set = {c.name.lower() for c in det_list}
            tp = len(gt_set & det_set)
            fp = len(det_set - gt_set)
            fn = len(gt_set - det_set)
            det_count = len(det_list)
        else:
            det_count = len(by_type.get(cat, []))
            tp = min(gt_count, det_count)
            fp = max(0, det_count - gt_count)
            fn = max(0, gt_count - det_count)

        p, r, f1 = _prf1(tp, fp, fn)
        per_category[cat] = CategoryMetrics(
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=p,
            recall=r,
            f1=f1,
            gt_count=gt_count,
            detected_count=det_count,
        )
        sum_tp += tp
        sum_fp += fp
        sum_fn += fn
        sum_gt += gt_count
        sum_det += det_count

    op, or_, of1 = _prf1(sum_tp, sum_fp, sum_fn)
    overall = OverallMetrics(
        true_positives=sum_tp,
        false_positives=sum_fp,
        false_negatives=sum_fn,
        precision=op,
        recall=or_,
        f1=of1,
        gt_count=sum_gt,
        detected_count=sum_det,
    )

    return BenchmarkResult(per_category=per_category, overall=overall)


def render_benchmark_result(
    result: BenchmarkResult,
    fmt: str,
    *,
    console: Console,
) -> None:
    if fmt == "json":
        payload = {
            "per_category": {k: v.model_dump() for k, v in result.per_category.items()},
            "overall": result.overall.model_dump(),
        }
        console.print(json.dumps(payload, indent=2))
        return
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(
            [
                "category",
                "true_positives",
                "false_positives",
                "false_negatives",
                "precision",
                "recall",
                "f1",
                "gt_count",
                "detected_count",
            ]
        )
        for cat, m in sorted(result.per_category.items()):
            w.writerow(
                [
                    cat,
                    m.true_positives,
                    m.false_positives,
                    m.false_negatives,
                    f"{m.precision:.6f}",
                    f"{m.recall:.6f}",
                    f"{m.f1:.6f}",
                    m.gt_count,
                    m.detected_count,
                ]
            )
        o = result.overall
        w.writerow(
            [
                "overall",
                o.true_positives,
                o.false_positives,
                o.false_negatives,
                f"{o.precision:.6f}",
                f"{o.recall:.6f}",
                f"{o.f1:.6f}",
                o.gt_count,
                o.detected_count,
            ]
        )
        console.print(buf.getvalue().rstrip())
        return

    table = Table(title="Benchmark (per category)")
    table.add_column("Category")
    table.add_column("TP", justify="right")
    table.add_column("FP", justify="right")
    table.add_column("FN", justify="right")
    table.add_column("P", justify="right")
    table.add_column("R", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("GT", justify="right")
    table.add_column("Det", justify="right")
    for cat, m in sorted(result.per_category.items()):
        table.add_row(
            cat,
            str(m.true_positives),
            str(m.false_positives),
            str(m.false_negatives),
            f"{m.precision:.3f}",
            f"{m.recall:.3f}",
            f"{m.f1:.3f}",
            str(m.gt_count),
            str(m.detected_count),
        )
    console.print(table)
    ot = Table(title="Overall")
    ot.add_column("TP", justify="right")
    ot.add_column("FP", justify="right")
    ot.add_column("FN", justify="right")
    ot.add_column("P", justify="right")
    ot.add_column("R", justify="right")
    ot.add_column("F1", justify="right")
    ot.add_column("GT", justify="right")
    ot.add_column("Det", justify="right")
    o = result.overall
    ot.add_row(
        str(o.true_positives),
        str(o.false_positives),
        str(o.false_negatives),
        f"{o.precision:.3f}",
        f"{o.recall:.3f}",
        f"{o.f1:.3f}",
        str(o.gt_count),
        str(o.detected_count),
    )
    console.print(ot)
