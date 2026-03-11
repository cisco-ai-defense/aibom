#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Benchmark script for cisco-aibom.

Clones public repositories from langchain-ai, crewAIInc, and other notable
AI/ML projects, runs the AIBOM analyzer against each, and produces a
consolidated benchmark report showing detection counts per category.

Usage:
    python benchmarks/run_benchmarks.py [--output benchmarks/results.json] [--skip-clone]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aibom.categorizer import categorize_symbols
from aibom.catalog_db import CatalogDB
from aibom.completeness import compute_completeness_score
from aibom.framework_config_parser import parse_project_configs
from aibom.cst_parser import parse_source_code
from aibom.db_loader import ensure_local_database
from aibom.js_parser import is_js_file, parse_js_source_code
from aibom.notebook_parser import extract_code_from_notebook
from aibom.structures import CodeAnalysisResult
from aibom.workflow_analyzer import build_workflow_index

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)

BENCHMARK_REPOS: List[Dict[str, str]] = [
    # ── langchain-ai ─────────────────────────────────────────────────────
    {"org": "langchain-ai", "repo": "langchain", "sparse": "libs/langchain/langchain"},
    {"org": "langchain-ai", "repo": "langgraph", "sparse": "libs/langgraph/langgraph"},
    {"org": "langchain-ai", "repo": "langchain-academy"},
    {"org": "langchain-ai", "repo": "memory-agent"},
    {"org": "langchain-ai", "repo": "rag-research-agent-template"},
    {"org": "langchain-ai", "repo": "react-agent"},
    {"org": "langchain-ai", "repo": "retrieval-agent-template"},
    {"org": "langchain-ai", "repo": "email-assistant"},
    {"org": "langchain-ai", "repo": "chat-langchain"},
    {"org": "langchain-ai", "repo": "langchain-benchmarks"},
    # ── crewAIInc ────────────────────────────────────────────────────────
    {"org": "crewAIInc", "repo": "crewAI-examples"},
    {"org": "crewAIInc", "repo": "crewAI"},
    # ── Notable community / reference repos ──────────────────────────────
    {"org": "langchain-ai", "repo": "opengpts"},
    {"org": "langchain-ai", "repo": "langserve"},
    # ── JS/TS AI repos ──────────────────────────────────────────────────
    {"org": "langchain-ai", "repo": "langchainjs", "sparse": "langchain/src"},
    {"org": "vercel", "repo": "ai", "sparse": "packages/ai/core"},
    {"org": "huggingface", "repo": "transformers.js", "sparse": "src"},
]


def clone_repo(
    org: str,
    repo: str,
    dest_dir: Path,
    sparse: Optional[str] = None,
) -> Optional[Path]:
    """Shallow-clone a GitHub repo into *dest_dir*/<repo>.

    If *sparse* is set, only that sub-path is checked out (useful for
    monorepos like langchain).
    """
    repo_dir = dest_dir / repo
    if repo_dir.exists():
        LOGGER.info("  [skip] %s/%s already cloned", org, repo)
        return repo_dir

    url = f"https://github.com/{org}/{repo}.git"
    LOGGER.info("  Cloning %s/%s ...", org, repo)

    try:
        if sparse:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", url, str(repo_dir)],
                check=True,
                capture_output=True,
                timeout=120,
            )
            subprocess.run(
                ["git", "sparse-checkout", "set", sparse],
                cwd=str(repo_dir),
                check=True,
                capture_output=True,
                timeout=30,
            )
        else:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(repo_dir)],
                check=True,
                capture_output=True,
                timeout=120,
            )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        LOGGER.warning("  Failed to clone %s/%s: %s", org, repo, exc)
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        return None

    return repo_dir


_SOURCE_EXTENSIONS = {".py", ".ipynb", ".js", ".jsx", ".ts", ".tsx"}
_EXCLUDE_DIRS = {".git", "__pycache__", "node_modules", ".tox", ".venv", "venv", "dist", "build", ".eggs"}


def find_files(path: Path) -> List[Path]:
    """Find all source files (Python, JS/TS, notebooks), excluding junk directories."""
    results: List[Path] = []
    for item in path.rglob("*"):
        if any(part in _EXCLUDE_DIRS for part in item.parts):
            continue
        if item.suffix in _SOURCE_EXTENSIONS and item.is_file():
            results.append(item)
    return results


def analyze_repo(
    repo_path: Path,
    connector: CatalogDB,
) -> Dict[str, Any]:
    """Run the full AIBOM analysis pipeline on a repo directory."""
    files = find_files(repo_path)
    py_files = [f for f in files if f.suffix == ".py"]
    nb_files = [f for f in files if f.suffix == ".ipynb"]
    js_files = [f for f in files if is_js_file(f)]

    analysis_results: List[CodeAnalysisResult] = []

    config_results = parse_project_configs(repo_path)
    analysis_results.extend(config_results)

    for py_file in py_files:
        try:
            source_code = py_file.read_text(encoding="utf-8")
            result = parse_source_code(str(py_file), source_code)
            analysis_results.append(result)
        except Exception:
            LOGGER.debug("Failed to parse %s", py_file, exc_info=True)

    for nb_file in nb_files:
        try:
            source_code = extract_code_from_notebook(nb_file)
            if source_code.strip():
                result = parse_source_code(str(nb_file), source_code)
                analysis_results.append(result)
        except Exception:
            LOGGER.debug("Failed to parse notebook %s", nb_file, exc_info=True)

    for js_file in js_files:
        try:
            source_code = js_file.read_text(encoding="utf-8")
            result = parse_js_source_code(str(js_file), source_code)
            analysis_results.append(result)
        except Exception:
            LOGGER.debug("Failed to parse JS/TS file %s", js_file, exc_info=True)

    workflow_index = None
    try:
        workflow_index = build_workflow_index(py_files)
    except Exception:
        LOGGER.debug("Failed to build workflow index", exc_info=True)

    output = categorize_symbols(analysis_results, connector, workflow_index=workflow_index)

    category_counts: Dict[str, int] = {}
    total = 0
    for category, components in output.components.items():
        count = len(components)
        category_counts[category] = count
        total += count

    relationship_counts: Dict[str, int] = {}
    for rel in output.relationships:
        relationship_counts[rel.label] = relationship_counts.get(rel.label, 0) + 1

    completeness_report = compute_completeness_score(output)

    return {
        "files_scanned": len(py_files) + len(nb_files) + len(js_files),
        "py_files": len(py_files),
        "nb_files": len(nb_files),
        "js_files": len(js_files),
        "config_files_parsed": len(config_results),
        "total_components": total,
        "categories": category_counts,
        "total_relationships": len(output.relationships),
        "relationship_types": relationship_counts,
        "completeness": completeness_report.to_dict(),
        "components_detail": {
            cat: [
                {
                    "name": c.get("name", ""),
                    "file_path": c.get("file_path", ""),
                    "line_number": c.get("line_number", 0),
                }
                for c in comps
            ]
            for cat, comps in output.components.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AIBOM Benchmark Runner")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results.json",
        help="Path to write benchmark results JSON.",
    )
    parser.add_argument(
        "--clone-dir",
        type=Path,
        default=Path(__file__).parent / "_repos",
        help="Directory to clone repos into.",
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Skip cloning; use existing repo directories.",
    )
    args = parser.parse_args()

    clone_dir: Path = args.clone_dir
    clone_dir.mkdir(parents=True, exist_ok=True)

    db_path = ensure_local_database()
    connector = CatalogDB(db_path)

    results: Dict[str, Any] = {
        "benchmark_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repos": {},
        "summary": {},
    }

    total_components_all = 0
    total_relationships_all = 0
    total_files_all = 0
    category_totals: Dict[str, int] = {}
    rel_totals: Dict[str, int] = {}

    for repo_info in BENCHMARK_REPOS:
        org = repo_info["org"]
        repo = repo_info["repo"]
        sparse = repo_info.get("sparse")
        key = f"{org}/{repo}"

        print(f"\n{'='*60}")
        print(f"  Benchmarking: {key}")
        print(f"{'='*60}")

        if not args.skip_clone:
            repo_path = clone_repo(org, repo, clone_dir, sparse=sparse)
        else:
            repo_path = clone_dir / repo
            if not repo_path.exists():
                LOGGER.warning("  Repo dir %s not found, skipping.", repo_path)
                repo_path = None

        if not repo_path:
            results["repos"][key] = {"error": "clone_failed"}
            continue

        try:
            repo_result = analyze_repo(repo_path, connector)
            results["repos"][key] = repo_result

            total_components_all += repo_result["total_components"]
            total_relationships_all += repo_result["total_relationships"]
            total_files_all += repo_result["files_scanned"]
            for cat, count in repo_result["categories"].items():
                category_totals[cat] = category_totals.get(cat, 0) + count
            for rel, count in repo_result["relationship_types"].items():
                rel_totals[rel] = rel_totals.get(rel, 0) + count

            print(f"  Files: {repo_result['files_scanned']}  |  Components: {repo_result['total_components']}  |  Relationships: {repo_result['total_relationships']}")
            for cat, count in sorted(repo_result["categories"].items()):
                print(f"    {cat}: {count}")
        except Exception as exc:
            LOGGER.error("  Error analyzing %s: %s", key, exc)
            results["repos"][key] = {"error": str(exc)}

    connector.close()

    results["summary"] = {
        "repos_benchmarked": len([r for r in results["repos"].values() if "error" not in r]),
        "repos_failed": len([r for r in results["repos"].values() if "error" in r]),
        "total_files_scanned": total_files_all,
        "total_components_detected": total_components_all,
        "total_relationships_detected": total_relationships_all,
        "category_totals": category_totals,
        "relationship_totals": rel_totals,
    }

    print(f"\n{'='*60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"  Repos benchmarked: {results['summary']['repos_benchmarked']}")
    print(f"  Total files scanned: {total_files_all}")
    print(f"  Total components: {total_components_all}")
    print(f"  Total relationships: {total_relationships_all}")
    print(f"  Categories:")
    for cat, count in sorted(category_totals.items()):
        print(f"    {cat}: {count}")
    print(f"  Relationships:")
    for rel, count in sorted(rel_totals.items()):
        print(f"    {rel}: {count}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Results written to: {args.output}")


if __name__ == "__main__":
    main()
