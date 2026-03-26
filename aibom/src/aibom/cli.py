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

import copy
import json
import logging
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich import box
from rich.console import Console

_LOGGER = logging.getLogger(__name__)
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from .report_sender import post_report_with_retries
from .utils.version import resolve_package_version
from .categorizer import categorize_symbols
from .framework_config_parser import parse_project_configs
from .container_utils import extract_app_from_docker, is_docker_image
from .cst_parser import parse_source_code
from .catalog_db import CatalogDB
from .custom_catalog import (
    CustomCatalogConfig,
    discover_custom_catalog,
    load_custom_catalog,
)
from .db_loader import ensure_local_database
from .notebook_parser import extract_code_from_notebook
from .structures import CodeAnalysisResult
from .api_handler import start_api_server
from .workflow_analyzer import build_workflow_index, workflow_identifier
from .reporters import get_reporter, reporter_registry
from .models.enums import Severity as SeverityEnum
from .kb.manager import KBManager, KBError

console = Console()

_VALID_OUTPUT_FORMATS = {"plaintext", "json", "api"} | {r.name for r in reporter_registry}

app = typer.Typer(
    help="Generate an AI BOM from source code.",
    no_args_is_help=True,
)

ENV_FILE_ENV_VAR = "AIBOM_ENV_FILE"


def _default_env_path() -> Optional[Path]:
    """Resolve a default .env using importlib.resources to avoid brittle __file__ math."""
    # Prefer a local .env in the current working directory.
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        return cwd_env

    try:
        pkg_root = importlib_resources.files(__package__.split(".")[0])
        # When running from source, the repo root is typically one level above the package.
        repo_root = Path(pkg_root).parent
        candidate = repo_root / ".env"
        if candidate.exists():
            return candidate
    except Exception:
        return None
    return None


ANALYZER_VERSION = resolve_package_version("cisco-aibom")


def _load_env_file() -> None:
    """Populate os.environ from a project-level .env file if it exists."""
    env_path = os.environ.get(ENV_FILE_ENV_VAR)
    candidate: Optional[Path] = None
    if env_path:
        candidate = Path(env_path).expanduser()
    else:
        candidate = _default_env_path()

    if not candidate or not candidate.exists():
        return

    try:
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or key in os.environ:
                continue
            if value and value[0] == value[-1] and value.startswith(("'", '"')):
                value = value[1:-1]
            os.environ[key] = value
    except OSError as exc:
        logging.debug("Failed to read env file %s: %s", candidate, exc)


_load_env_file()


def _utcnow_iso() -> str:
    """Return current UTC timestamp in ISO 8601 with Z suffix."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _print_timing_table(console: "rich.console.Console", result: "PipelineResult") -> None:  # type: ignore[name-defined]
    from rich.table import Table

    from .scanners import scanner_timings

    table = Table(title="Pipeline Timing", show_footer=True)
    table.add_column("Stage", footer="Total")
    table.add_column("Elapsed", justify="right", footer=f"{result.total_elapsed_s:.2f}s")
    table.add_column("%", justify="right")
    table.add_column("Detail")

    for st in result.timings:
        pct = (st.elapsed_s / result.total_elapsed_s * 100) if result.total_elapsed_s else 0
        table.add_row(st.name, f"{st.elapsed_s:.2f}s", f"{pct:.1f}%", st.detail)

    console.print(table)

    if scanner_timings:
        sc_table = Table(title="Per-Scanner Timing")
        sc_table.add_column("Scanner")
        sc_table.add_column("Elapsed", justify="right")
        sc_table.add_column("%", justify="right")

        scan_stage = next((t for t in result.timings if t.name == "scan"), None)
        scan_total = scan_stage.elapsed_s if scan_stage else 1.0
        for name, elapsed in sorted(scanner_timings.items(), key=lambda x: -x[1]):
            pct = (elapsed / scan_total * 100) if scan_total else 0
            table_style = "bold red" if pct > 30 else ""
            sc_table.add_row(name, f"{elapsed:.2f}s", f"{pct:.1f}%", style=table_style)

        console.print(sc_table)


def _record_analysis_error(
    run_errors: List[Dict[str, Any]],
    source_summary: Optional[Dict[str, Any]],
    source: str,
    message: str,
    *,
    file_path: Optional[str] = None,
    severity: str = "warning",
) -> None:
    """Track errors so they surface in the JSON report."""
    entry: Dict[str, Any] = {
        "source": source,
        "message": message,
        "severity": severity,
    }
    if file_path:
        entry["file_path"] = file_path
    run_errors.append(entry)
    if source_summary is None:
        return
    source_summary["errors"].append(entry)
    source_summary["status_detail"] = message
    if severity == "fatal":
        source_summary["status"] = "failed"
    elif source_summary.get("status") not in {"failed", "completed_with_errors"}:
        source_summary["status"] = "completed_with_errors"


@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context) -> None:
    """Ensure a subcommand is provided when invoking the CLI."""
    if ctx.invoked_subcommand is None:
        console.print(
            "[bold red]No subcommand provided.[/] Please use [green]analyze[/], [green]report[/], or [green]kb[/]."
        )
        raise typer.Exit(code=1)


def _build_scan_result(
    all_analysis_outputs: Dict[str, Any],
    run_metadata: Dict[str, Any],
    run_errors: List[Dict[str, Any]],
) -> "ScanResult":
    """Bridge legacy CategorizationOutput to the v2 ScanResult model."""
    from .models import (
        AIComponent as V2Component,
        AIComponentType,
        ComponentRelationship as V2Relationship,
        ScanResult,
        SourceResult,
    )

    _TYPE_MAP = {
        "model": AIComponentType.MODEL,
        "agent": AIComponentType.AGENT,
        "tool": AIComponentType.TOOL,
        "mcp_server": AIComponentType.MCP_SERVER,
        "mcp_client": AIComponentType.MCP_CLIENT,
        "embedding": AIComponentType.EMBEDDING,
        "vector_store": AIComponentType.VECTOR_STORE,
        "datastore": AIComponentType.VECTOR_STORE,
        "dataset": AIComponentType.DATASET,
        "prompt": AIComponentType.PROMPT,
        "guardrail": AIComponentType.GUARDRAIL,
        "observability": AIComponentType.OBSERVABILITY,
        "memory": AIComponentType.MEMORY,
        "retriever": AIComponentType.RETRIEVER,
    }

    sources = []
    for source_path, output in all_analysis_outputs.items():
        categorized = getattr(output, "components", output)
        relationships = getattr(output, "relationships", [])

        components: List[V2Component] = []
        for category, items in categorized.items():
            comp_type = _TYPE_MAP.get(category, AIComponentType.OTHER)
            for item in (items or []):
                components.append(V2Component(
                    name=item.get("name", "unknown"),
                    component_type=comp_type,
                    file_path=item.get("file_path", ""),
                    line_number=item.get("line_number", 0),
                    framework=item.get("framework", ""),
                    model_name=item.get("model_name"),
                    embedding_model=item.get("embedding_model"),
                    description=item.get("description"),
                    text=item.get("text"),
                    instance_id=item.get("instance_id", ""),
                    metadata={
                        k: v for k, v in item.items()
                        if k not in {
                            "name", "file_path", "line_number", "framework",
                            "model_name", "embedding_model", "description",
                            "text", "instance_id", "category",
                        }
                    },
                ))

        v2_rels: List[V2Relationship] = []
        for rel in relationships:
            v2_rels.append(V2Relationship(
                source_instance_id=getattr(rel, "source_instance_id", ""),
                target_instance_id=getattr(rel, "target_instance_id", ""),
                label=getattr(rel, "label", ""),
                source_name=getattr(rel, "source_name", ""),
                target_name=getattr(rel, "target_name", ""),
            ))

        sources.append(SourceResult(
            path=source_path,
            components=components,
            relationships=v2_rels,
        ))

    return ScanResult(
        metadata=run_metadata,
        sources=sources,
        errors=[e.get("message", str(e)) for e in run_errors],
    )


def _render_component_table(source: str, categorized_components: Dict[str, List[Dict[str, Any]]]) -> None:
    table = Table(
        "Category",
        "Count",
        "Total Workflows",
        title=f"Components in {source}",
        box=box.SIMPLE_HEAVY,
        header_style="bold magenta",
    )
    has_rows = False
    for category, components in sorted(categorized_components.items()):
        if not components:
            continue
        total_workflows = sum(len(comp.get("workflows") or []) for comp in components)
        table.add_row(category, str(len(components)), str(total_workflows))
        has_rows = True
    if has_rows:
        console.print(table)
    else:
        console.print(Panel.fit("No components detected.", title=source, style="yellow"))


def _build_workflow_tree(component: Dict[str, Any]) -> Tree:
    title = f"[bold]{component.get('name')}[/] ({component.get('category')})"
    root = Tree(title)
    workflows = sorted(component.get("workflows") or [], key=lambda wf: wf.get("distance", 0))
    for workflow in workflows:
        distance = workflow.get("distance", 0)
        func = workflow.get("function", "unknown")
        location = f"{workflow.get('file_path', '')}:{workflow.get('line', '')}"
        callsite = workflow.get("callsite_line")
        arguments = workflow.get("call_arguments")
        node_label = f"[cyan]{func}[/] • distance={distance} • [dim]{location}[/]"
        if callsite:
            node_label += f" • call at line {callsite}"
        if arguments:
            node_label += f" • args={arguments}"
        root.add(node_label)
    return root


def _render_relationship_table(relationships: List[Any]) -> None:
    if not relationships:
        return
    table = Table(
        "Source",
        "Label",
        "Target",
        title="Derived Relationships",
        header_style="bold blue",
        box=box.MINIMAL_DOUBLE_HEAD,
    )
    for rel in relationships:
        if isinstance(rel, dict):
            source_name = rel.get("source_name", "")
            source_category = rel.get("source_category", "")
            target_name = rel.get("target_name", "")
            target_category = rel.get("target_category", "")
            label = rel.get("label", "")
        else:
            source_name = getattr(rel, "source_name", "")
            source_category = getattr(rel, "source_category", "")
            target_name = getattr(rel, "target_name", "")
            target_category = getattr(rel, "target_category", "")
            label = getattr(rel, "label", "")
        table.add_row(
            f"{source_name} ({source_category})",
            label,
            f"{target_name} ({target_category})",
        )
    console.print(table)


def _display_v2_summary(source: str, components: list, relationships: list) -> None:
    """Rich summary for v2 detector output."""
    from collections import Counter
    panel_title = f"[bold green]Analysis Summary (v2)[/] • {source}"
    console.print(Panel(panel_title, style="green", expand=False))
    if not components:
        console.print(Panel.fit("No AI components detected.", title=source, style="yellow"))
        return
    by_type: Counter = Counter()
    grouped: Dict[str, list] = {}
    for comp in components:
        ctype = comp.component_type.value if hasattr(comp.component_type, "value") else str(comp.component_type)
        by_type[ctype] += 1
        grouped.setdefault(ctype, []).append(comp)
    table = Table("Category", "Count", title=f"Components in {source}", box=box.SIMPLE_HEAVY, header_style="bold magenta")
    for ctype, count in sorted(by_type.items()):
        table.add_row(ctype, str(count))
    console.print(table)
    for ctype, items in sorted(grouped.items()):
        console.print(Panel.fit(f"{ctype.upper()} details", style="bold white"))
        for comp in items[:5]:
            loc = f"{comp.file_path}:{comp.line_path}" if hasattr(comp, "line_path") else f"{comp.file_path}:{comp.line_number}"
            extra = ""
            if comp.model_name:
                extra = f" model={comp.model_name}"
            elif comp.framework:
                extra = f" framework={comp.framework}"
            provider = comp.metadata.get("provider", "")
            if provider and provider != "unknown":
                extra += f" provider={provider}"
            console.print(f"  [cyan]{comp.name}[/]{extra} [dim]{loc}[/]")


def _display_analysis_summary(all_analysis_outputs: Dict[str, Any], max_examples: int = 3) -> None:
    for source, output in all_analysis_outputs.items():
        if isinstance(output, dict) and output.get("_v2"):
            _display_v2_summary(source, output["components"], output["relationships"])
            continue
        categorized_components = getattr(output, "components", output)
        panel_title = f"[bold green]Analysis Summary[/] • {source}"
        console.print(Panel(panel_title, style="green", expand=False))
        _render_component_table(source, categorized_components)
        example_sections = 0
        for category, components in categorized_components.items():
            if not components:
                continue
            console.print(Panel.fit(f"{category.upper()} details", style="bold white"))
            for component in components[:max_examples]:
                console.print(_build_workflow_tree(component))
            example_sections += 1
        relationships = getattr(output, "relationships", [])
        _render_relationship_table(relationships)
        console.print()  # spacing


def _find_python_files(path: Path) -> List[Path]:
    """Finds all .py and .ipynb files in a given path (file or directory)."""
    if path.is_file() and path.suffix in (".py", ".ipynb"):
        return [path]
    if path.is_dir():
        py_files = list(path.rglob("*.py"))
        nb_files = list(path.rglob("*.ipynb"))
        return py_files + nb_files
    return []


def _generate_plaintext_report(all_analysis_outputs, output_file: Path):
    """Generate plaintext report format."""
    report_lines = ["--- AI BOM Analysis Report ---"]
    grand_total = 0
    for source, output in all_analysis_outputs.items():
        categorized_components = getattr(output, "components", output)
        report_lines.append(f"\n\n--- Results for source: {source} ---")
        total_components = 0
        for category, components in categorized_components.items():
            if components:
                total_components += len(components)
                report_lines.append(f"\n[+] Found {len(components)} {category.upper()}:")
                for comp in components:
                    report_lines.append(f"  - Name: {comp['name']}")
                    if 'text' in comp:
                        prompt_text = json.dumps(comp['text'], indent=4)
                        report_lines.append(f"    Text: {prompt_text}")
                    if 'model_name' in comp:
                        report_lines.append(f"    Model: {comp['model_name']}")
                    if 'embedding_model' in comp:
                        report_lines.append(f"    Embedding Model: {comp['embedding_model']}")
                    report_lines.append(f"    Source: {comp['file_path']}:{comp['line_number']}")
                    workflows = comp.get('workflows')
                    if workflows:
                        report_lines.append("    Workflows:")
                        for wf in workflows:
                            workflow_name = wf.get('function', 'unknown')
                            wf_file = wf.get('file_path', '')
                            wf_line = wf.get('line', '')
                            distance = wf.get('distance', 0)
                            report_lines.append(
                                f"      - {workflow_name} (distance {distance}) [{wf_file}:{wf_line}]"
                            )

        if total_components == 0:
            report_lines.append("No known AI components were found in this source.")
        relationships = getattr(output, "relationships", [])
        if relationships:
            report_lines.append(f"\n[+] Derived Relationships ({len(relationships)}):")
            for rel in relationships:
                report_lines.append(
                    f"  - {rel.source_name} [{rel.source_category}] {rel.label} {rel.target_name} [{rel.target_category}]"
                )
        grand_total += total_components

    if grand_total == 0:
        report_lines.append("\nNo known AI components were found in any of the specified sources.")
    else:
        report_lines.append(f"\n--- End of Report: Found {grand_total} total components across all sources. ---")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    logging.info(f"Plaintext report written to {output_file}")


def _convert_to_container_path(file_path: str, temp_dir: str) -> str:
    """Convert host temporary directory path to container path."""
    if not temp_dir or temp_dir not in file_path:
        return file_path
    
    # Convert temp_dir/app/... -> /app/...
    if "/app/" in file_path:
        # Find the /app/ part and everything after it
        app_index = file_path.find("/app/")
        return file_path[app_index:]
    
    # For other paths, try to extract meaningful container path
    # Remove the temp directory prefix
    relative_path = file_path.replace(temp_dir, "")
    if relative_path.startswith("/"):
        relative_path = relative_path[1:]
    
    # If it looks like site-packages content, map to container path
    if "site-packages" in relative_path:
        # Extract the part after site-packages-X/
        parts = relative_path.split("/")
        if len(parts) > 1 and parts[0].startswith("site-packages"):
            return "/" + "/".join(parts[1:])
    
    # Default: assume it's in /app
    return "/app/" + relative_path if relative_path else "/app"


def _convert_paths_in_output(analysis_output, temp_dir: str):
    """Convert all file paths in components from host paths to container paths."""
    if not temp_dir:
        return analysis_output
    
    categorized_components = getattr(analysis_output, "components", analysis_output)
    for category, components in categorized_components.items():
        for component in components:
            if "file_path" in component:
                component["file_path"] = _convert_to_container_path(component["file_path"], temp_dir)
    
    return analysis_output


def _map_source_kind(kind: Optional[str]) -> str:
    """Map internal source kind strings to the API enum expected by the backend."""
    normalized = (kind or "").replace("_", "-").lower()
    if normalized == "local-path":
        return "SOURCE_KIND_LOCAL_PATH"
    if normalized == "container":
        return "SOURCE_KIND_CONTAINER"
    return "SOURCE_KIND_UNSPECIFIED"


def _build_submission_payload(
    report: Dict[str, Any],
    source_outcomes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Wrap the generated report in the API submission envelope."""
    analysis = report.get("aibom_analysis", {})
    metadata = analysis.get("metadata", {})

    source_kinds = {
        _map_source_kind(info.get("source_kind"))
        for info in source_outcomes.values()
        if info.get("source_kind")
    }
    source_kind = "SOURCE_KIND_UNSPECIFIED"
    if len(source_kinds) == 1:
        source_kind = source_kinds.pop()

    sources_payload: List[Dict[str, str]] = []
    for source, info in source_outcomes.items():
        source_name = info.get("source_name") or Path(source).name or str(source)
        source_path = info.get("source_path") or source
        sources_payload.append(
            {
                "name": str(source_name),
                "path": str(source_path),
            }
        )

    submitted_at = (
        metadata.get("completed_at")
        or metadata.get("started_at")
        or _utcnow_iso()
    )
    return {
        "run_id": metadata.get("run_id"),
        "analyzer_version": metadata.get("analyzer_version") or ANALYZER_VERSION,
        "submitted_at": submitted_at,
        "source_kind": source_kind,
        "sources": sources_payload,
        "report": report,
    }


def _generate_json_report(
    all_analysis_outputs,
    output_file: Path,
    metadata: Optional[Dict[str, Any]] = None,
    source_summaries: Optional[Dict[str, Dict[str, Any]]] = None,
    run_errors: Optional[List[Dict[str, Any]]] = None,
):
    """Generate JSON report format."""
    metadata = metadata or {}
    source_summaries = source_summaries or {}
    total_sources = len(source_summaries) if source_summaries else len(all_analysis_outputs)
    report = {
        "aibom_analysis": {
            "metadata": metadata,
            "sources": {},
            "summary": {
                "total_sources": total_sources,
                "total_components": 0,
                "categories": {},
                "total_relationships": 0,
                "total_workflows": 0,
            },
            "errors": run_errors or [],
        }
    }
    
    category_totals = {}
    grand_total = 0
    grand_relationships = 0
    grand_workflows = 0
    
    for source, output in all_analysis_outputs.items():
        categorized_components = getattr(output, "components", output)
        source_data = {
            "components": {},
            "total_components": 0,
            "workflows": [],
            "total_workflows": 0,
        }
        workflow_catalog: Dict[str, Dict[str, Any]] = {}

        source_total = 0
        for category, components in categorized_components.items():
            if not components:
                continue
            source_total += len(components)
            category_totals[category] = category_totals.get(category, 0) + len(components)

            category_components: List[Dict[str, Any]] = []
            for component in components:
                component_copy = copy.deepcopy(component)
                workflows = component_copy.get("workflows") or []
                if workflows:
                    enriched_workflows: List[Dict[str, Any]] = []
                    for workflow in workflows:
                        wf_copy = copy.deepcopy(workflow)
                        wf_id = workflow_identifier(
                            wf_copy.get("function"),
                            wf_copy.get("file_path"),
                            wf_copy.get("line"),
                        )
                        wf_copy["id"] = wf_id
                        enriched_workflows.append(wf_copy)

                        existing = workflow_catalog.get(wf_id)
                        wf_distance = wf_copy.get("distance", 0)
                        if not existing:
                            workflow_catalog[wf_id] = {
                                "id": wf_id,
                                "function": wf_copy.get("function"),
                                "file_path": wf_copy.get("file_path"),
                                "line": wf_copy.get("line"),
                                "distance": wf_distance,
                            }
                        else:
                            existing["distance"] = min(existing.get("distance", wf_distance), wf_distance)
                    component_copy["workflows"] = enriched_workflows
                category_components.append(component_copy)
            source_data["components"][category] = category_components

        source_data["total_components"] = source_total
        source_data["workflows"] = list(workflow_catalog.values())
        source_data["total_workflows"] = len(workflow_catalog)
        grand_workflows += len(workflow_catalog)
        relationships = getattr(output, "relationships", [])
        if relationships:
            source_data["relationships"] = [rel.to_dict() for rel in relationships]
            grand_relationships += len(relationships)
        summary_payload = source_summaries.get(source, {})
        source_data["summary"] = {
            "status": summary_payload.get("status", "completed"),
            "status_detail": summary_payload.get("status_detail"),
            "source_kind": summary_payload.get("source_kind"),
            "assets_discovered": summary_payload.get("assets_discovered", source_total),
            "branches_scanned": summary_payload.get("branches_scanned"),
            "last_generated_at": summary_payload.get("last_generated_at"),
        }
        if summary_payload.get("errors"):
            source_data["summary"]["errors"] = summary_payload["errors"]
        report["aibom_analysis"]["sources"][source] = source_data
        grand_total += source_total
    
    # Ensure sources that failed prior to component extraction are still represented.
    for missing_source, summary_payload in source_summaries.items():
        if missing_source in report["aibom_analysis"]["sources"]:
            continue
        placeholder = {
            "components": {},
            "total_components": summary_payload.get("assets_discovered", 0),
            "workflows": [],
            "total_workflows": 0,
            "summary": {
                "status": summary_payload.get("status", "failed"),
                "status_detail": summary_payload.get("status_detail"),
                "source_kind": summary_payload.get("source_kind"),
                "assets_discovered": summary_payload.get("assets_discovered", 0),
                "branches_scanned": summary_payload.get("branches_scanned"),
                "last_generated_at": summary_payload.get("last_generated_at"),
            },
        }
        if summary_payload.get("errors"):
            placeholder["summary"]["errors"] = summary_payload["errors"]
        report["aibom_analysis"]["sources"][missing_source] = placeholder

    report["aibom_analysis"]["summary"]["total_components"] = grand_total
    report["aibom_analysis"]["summary"]["total_relationships"] = grand_relationships
    report["aibom_analysis"]["summary"]["categories"] = category_totals
    report["aibom_analysis"]["summary"]["total_workflows"] = grand_workflows
    status_counts: Dict[str, int] = {}
    for info in source_summaries.values():
        status = info.get("status", "completed")
        status_counts[status] = status_counts.get(status, 0) + 1
    if status_counts:
        report["aibom_analysis"]["summary"]["status_counts"] = status_counts
    if metadata.get("status"):
        report["aibom_analysis"]["summary"]["status"] = metadata["status"]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    logging.info(f"JSON report written to {output_file}")
    return report


def _render_json_report_console(report: Dict[str, Any]) -> None:
    analysis = report.get("aibom_analysis")
    if not analysis:
        raise ValueError("Report does not contain an 'aibom_analysis' key")
    summary = analysis.get("summary", {})
    summary_table = Table(
        "Total Sources",
        "Components",
        "Workflows",
        "Relationships",
        title="Report Summary",
        box=box.SIMPLE_HEAVY,
        header_style="bold green",
    )
    summary_table.add_row(
        str(summary.get("total_sources", 0)),
        str(summary.get("total_components", 0)),
        str(summary.get("total_workflows", 0)),
        str(summary.get("total_relationships", 0)),
    )
    console.print(summary_table)

    for source, source_data in analysis.get("sources", {}).items():
        console.print(Panel(f"[bold]{source}[/]", style="cyan"))
        components = source_data.get("components", {})
        _render_component_table(source, components)
        for category, category_components in components.items():
            if not category_components:
                continue
            console.print(f"[bold]{category.upper()}[/]")
            for component in category_components:
                console.print(_build_workflow_tree(component))
        _render_relationship_table(source_data.get("relationships", []))


@app.command("report")
def report_command(
    report_file: Path = typer.Argument(..., exists=True, readable=True, dir_okay=False, help="Path to a JSON report file."),
    raw: bool = typer.Option(False, "--raw-json", help="Display the raw JSON using syntax highlighting before the summary."),
) -> None:
    """Render a previously generated JSON report using Rich components."""
    try:
        data = json.loads(report_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        console.print(f"[red]Failed to parse JSON:[/] {exc}")
        raise typer.Exit(code=1)

    if raw:
        console.print(Syntax(json.dumps(data, indent=2), "json", theme="monokai"))
    _render_json_report_console(data)


@app.command("analyze")
def analyze(
    sources: Optional[List[str]] = typer.Argument(
        None, help="A list of source directories or container images to analyze."
    ),
    images_file: Optional[Path] = typer.Option(
        None,
        "--images-file",
        "-f",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to a JSON file containing a list of container images to scan.",
    ),
    output_format: str = typer.Option(
        "plaintext",
        "--output-format",
        "-o",
        help=(
            "Output format: plaintext, json, api, cyclonedx, sarif, spdx, "
            "html, markdown, csv, junit"
        ),
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-O",
        help="Path to write the report to (for json and plaintext formats).",
        writable=True,
        resolve_path=True,
    ),
    post_url: Optional[str] = typer.Option(
        None,
        "--post-url",
        help="Optional HTTP endpoint to POST the JSON report to (can also be set via AIBOM_POST_URL).",
        envvar="AIBOM_POST_URL",
    ),
    ai_defense_api_key: Optional[str] = typer.Option(
        None,
        "--ai-defense-api-key",
        help="API key sent as X-API-Key when POSTing the report to AI Defense endpoints.",
        envvar="AI_DEFENSE_API_KEY",
    ),
    post_timeout: float = typer.Option(
        30.0,
        "--post-timeout",
        help="Timeout (seconds) for posting the JSON report.",
        envvar="AIBOM_POST_TIMEOUT",
    ),
    post_verify_tls: bool = typer.Option(
        True,
        "--post-verify-tls/--no-post-verify-tls",
        help="Verify TLS certificates when POSTing the report.",
        envvar="AIBOM_POST_VERIFY_TLS",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    ),
    llm_model: Optional[str] = typer.Option(
        None,
        "--llm-model",
        help=(
            "LLM model name (e.g. gpt-4o, claude-sonnet-4-20250514).  "
            "In v2 mode this enables agentic enrichment after the "
            "deterministic scan (requires 'cisco-aibom[agentic]')."
        ),
    ),
    llm_api_key: Optional[str] = typer.Option(
        None,
        "--llm-api-key",
        help="LLM API key. May be optional for local LLM",
    ),
    llm_api_base: Optional[str] = typer.Option(
        None,
        "--llm-api-base",
        help="LLM API base URL",
    ),
    llm_api_version: Optional[str] = typer.Option(
        None,
        "--llm-api-version",
        help="LLM API version (for Azure OpenAI or some providers). May be optional for local LLM",
    ),
    show_summary: bool = typer.Option(
        True,
        "--show-summary/--no-show-summary",
        help="Display a Rich summary of the analysis results in the terminal.",
    ),
    custom_catalog: Optional[Path] = typer.Option(
        None,
        "--custom-catalog",
        help=(
            "Path to a custom catalog file (.aibom.yaml, .aibom.yml, or .aibom.json) "
            "that registers user-defined AI components, base-class rules, excludes, "
            "and relationship hints.  If not provided, auto-discovers "
            ".aibom.yaml/.yml/.json in each source directory."
        ),
        exists=False,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    fail_on: Optional[str] = typer.Option(
        None,
        "--fail-on",
        help=(
            "Exit with non-zero code if risk severity meets or exceeds this "
            "threshold: critical, high, medium, low."
        ),
    ),
    min_severity: str = typer.Option(
        "info",
        "--severity",
        help="Minimum severity of findings to include in the report.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Only emit high-confidence detections; suppress items that need agentic reasoning.",
    ),
    timing: bool = typer.Option(
        False,
        "--timing",
        help="Print a per-stage and per-scanner timing breakdown after analysis.",
    ),
    agentic_scope: str = typer.Option(
        "candidates",
        "--agentic-scope",
        help=(
            "Which components to send to the LLM for agentic enrichment. "
            "'candidates' (default) sends only needs_agentic items; "
            "'all' sends every component."
        ),
    ),
    agentic_batch_size: int = typer.Option(
        5,
        "--agentic-batch-size",
        help="Max components per agentic LLM invocation (default 5).",
    ),
    agentic_concurrency: int = typer.Option(
        1,
        "--agentic-concurrency",
        help="Max parallel agentic LLM batches (default 1, sequential).",
    ),
    agentic_fast_model: Optional[str] = typer.Option(
        None,
        "--agentic-fast-model",
        help=(
            "Cheaper/faster LLM for simple confirmations (model lookups, "
            "dependency checks). Falls back to --llm-model if not set."
        ),
    ),
    cache_dir: Optional[Path] = typer.Option(
        None,
        "--cache-dir",
        help=(
            "Directory for caching scan results keyed by repo@commit_sha. "
            "Repeated scans of the same codebase at the same revision are instant. "
            "Use 'cisco-aibom cache clear' to purge."
        ),
    ),
    validate: bool = typer.Option(
        False,
        "--validate",
        help="Validate the output against the format's schema and report errors.",
    ),
    repos_file: Optional[Path] = typer.Option(
        None,
        "--repos-file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help=(
            "Path to a file listing repo paths or git URLs "
            "(JSON array or newline-delimited text)."
        ),
    ),
    discover_repos: bool = typer.Option(
        False,
        "--discover-repos",
        help=(
            "Treat each positional source as a parent directory and "
            "auto-discover all git repositories underneath."
        ),
    ),
    github_org: Optional[str] = typer.Option(
        None,
        "--github-org",
        help="Discover and scan repos from a GitHub org/user.",
        envvar="AIBOM_GITHUB_ORG",
    ),
    gitlab_group: Optional[str] = typer.Option(
        None,
        "--gitlab-group",
        help="Discover and scan repos from a GitLab group.",
        envvar="AIBOM_GITLAB_GROUP",
    ),
    bitbucket_project: Optional[str] = typer.Option(
        None,
        "--bitbucket-project",
        help="Discover and scan repos from a Bitbucket workspace/project.",
        envvar="AIBOM_BITBUCKET_PROJECT",
    ),
    platform_token: Optional[str] = typer.Option(
        None,
        "--platform-token",
        help="Auth token for GitHub/GitLab/Bitbucket API access.",
        envvar="AIBOM_PLATFORM_TOKEN",
    ),
    repo_name_filter: Optional[str] = typer.Option(
        None,
        "--repo-filter",
        help="Filter discovered repos by name substring.",
    ),
    repo_topic_filter: Optional[str] = typer.Option(
        None,
        "--repo-topic",
        help="Filter discovered repos by topic/tag.",
    ),
    max_repos: Optional[int] = typer.Option(
        None,
        "--max-repos",
        help=(
            "Maximum number of repos to scan when using --discover-repos, "
            "--github-org, --gitlab-group, or --repos-file.  Repos are sorted "
            "by last-push date (most recent first)."
        ),
    ),
    parallel_repos: int = typer.Option(
        1,
        "--parallel-repos",
        help=(
            "Number of repositories to scan in parallel (default 1 = sequential).  "
            "Higher values speed up org-scale scans but require more memory."
        ),
    ),
    legacy_mode: bool = typer.Option(
        False,
        "--legacy-mode",
        help=(
            "Use the v1 KB-symbol-matching pipeline instead of the v2 "
            "targeted detectors.  Intended for backward compatibility only."
        ),
    ),
):
    """Analyzes a Python codebase to generate an AI BOM."""
    # Configure logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        logging.error(f"Invalid log level: {log_level}")
        raise typer.Exit(code=1)
    
    # Remove existing handlers and configure root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=numeric_level, format='%(levelname)s: %(message)s')

    # Validate output format
    if output_format not in _VALID_OUTPUT_FORMATS:
        valid = ", ".join(sorted(_VALID_OUTPUT_FORMATS))
        logging.error(f"Invalid output format '{output_format}'. Must be one of: {valid}")
        raise typer.Exit(code=1)

    if agentic_scope not in ("candidates", "all"):
        logging.error(
            f"Invalid --agentic-scope '{agentic_scope}'. Must be: candidates, all"
        )
        raise typer.Exit(code=1)

    # Validate severity options
    fail_on_severity: Optional[SeverityEnum] = None
    if fail_on:
        try:
            fail_on_severity = SeverityEnum(fail_on.lower())
        except ValueError:
            logging.error(f"Invalid --fail-on value '{fail_on}'. Must be: critical, high, medium, low, info")
            raise typer.Exit(code=1)
    try:
        severity_filter = SeverityEnum(min_severity.lower())
    except ValueError:
        logging.error(f"Invalid --severity value '{min_severity}'. Must be: critical, high, medium, low, info")
        raise typer.Exit(code=1)
    
    llm_config = None
    if llm_model:
        llm_config = {
            "model": llm_model,
            "api_key": llm_api_key,
            "api_base": llm_api_base,
            "api_version": llm_api_version,
        }
    
    sources_to_process = list(sources) if sources else []
    if images_file:
        try:
            with open(images_file, 'r') as f:
                images_from_file = json.load(f)
                if isinstance(images_from_file, list):
                    sources_to_process.extend(images_from_file)
                else:
                    logging.warning(f"Expected a JSON array in {images_file}, but found {type(images_from_file)}. Skipping.")
        except json.JSONDecodeError:
            logging.error(f"Could not decode JSON from {images_file}")
            raise typer.Exit(code=1)

    if repos_file:
        from .multi_repo import read_repos_file
        sources_to_process.extend(read_repos_file(repos_file))

    if discover_repos:
        from .multi_repo import discover_repos as _discover
        expanded: list[str] = []
        for src in sources_to_process:
            p = Path(src)
            if p.is_dir() and not (p / ".git").exists():
                repos = _discover(p)
                if repos:
                    console.print(
                        f"  [dim]Discovered {len(repos)} repo(s) under {src}[/]"
                    )
                    expanded.extend(str(r) for r in repos)
                else:
                    expanded.append(src)
            else:
                expanded.append(src)
        sources_to_process = expanded

    _platform_pairs: list[tuple[str, str]] = []
    if github_org:
        _platform_pairs.append(("github", github_org))
    if gitlab_group:
        _platform_pairs.append(("gitlab", gitlab_group))
    if bitbucket_project:
        _platform_pairs.append(("bitbucket", bitbucket_project))

    if _platform_pairs:
        from .platform_adapters import get_adapter

        for plat, ns in _platform_pairs:
            try:
                adapter = get_adapter(plat, token=platform_token)
                repos = adapter.list_repos(
                    ns,
                    name_filter=repo_name_filter,
                    topic_filter=repo_topic_filter,
                )
                console.print(
                    f"  [dim]{plat}: discovered {len(repos)} repo(s) "
                    f"in {ns}[/]"
                )
                sources_to_process.extend(r.clone_url for r in repos)
            except Exception as exc:
                console.print(
                    f"[yellow]Warning: {plat} discovery failed: {exc}[/]"
                )

    if len(sources_to_process) > 1 and llm_model:
        from .repo_triage import RepoTriager

        triage_llm_cfg = {"model": llm_model}
        if llm_base:
            triage_llm_cfg["api_base"] = llm_base
        if llm_key:
            triage_llm_cfg["api_key"] = llm_key

        triager = RepoTriager(llm_config=triage_llm_cfg)
        triage_results = triager.triage_repos(sources_to_process)

        deep = [t.repo_path for t in triage_results if t.decision == "deep-scan"]
        clone = [t.repo_path for t in triage_results if t.decision == "needs-clone"]
        skipped = [t.repo_path for t in triage_results if t.decision == "skip"]

        if skipped:
            console.print(
                f"  [dim]Repo triage: skipping {len(skipped)} non-AI repo(s)[/]"
            )
            for t in triage_results:
                if t.decision == "skip":
                    _LOGGER.info("Triage skip: %s — %s (%s)", t.repo_path, t.reason, t.method)

        sources_to_process = deep + clone

    if max_repos and len(sources_to_process) > max_repos:
        console.print(
            f"  [dim]Limiting to {max_repos} of {len(sources_to_process)} "
            f"discovered repos (--max-repos)[/]"
        )
        sources_to_process = sources_to_process[:max_repos]

    if parallel_repos > 1 and len(sources_to_process) > 1:
        console.print(
            f"  [dim]Scanning {len(sources_to_process)} repos with "
            f"--parallel-repos={parallel_repos}[/]"
        )

    if not sources_to_process:
        logging.error("No sources provided. Please specify a path or an images file.")
        raise typer.Exit(code=1)

    if output_format in ["plaintext", "json"] and not output_file:
        logging.error(f"--output-file is required for '{output_format}' format.")
        raise typer.Exit(code=1)

    all_analysis_outputs = {}
    run_errors: List[Dict[str, Any]] = []
    source_outcomes: Dict[str, Dict[str, Any]] = {}
    run_metadata: Dict[str, Any] = {
        "run_id": str(uuid.uuid4()),
        "analyzer_version": ANALYZER_VERSION,
        "started_at": _utcnow_iso(),
        "output_format": output_format,
        "sources_requested": len(sources_to_process),
    }
    db_path: Optional[Path] = None
    if legacy_mode:
        try:
            db_path = ensure_local_database(console=console)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[bold red]Knowledge base error:[/] {exc}")
            raise typer.Exit(code=1)

    explicit_config: Optional[CustomCatalogConfig] = None
    if custom_catalog:
        explicit_config = load_custom_catalog(Path(custom_catalog))

    clone_managers: list[Any] = []

    for source in sources_to_process:
        console.print(Panel.fit(f"Analyzing Source: {source}", style="bold cyan"))
        temp_dir = None
        clone_ctx = None

        from .multi_repo import is_git_url, ClonedRepo

        if is_git_url(source):
            try:
                clone_ctx = ClonedRepo(source)
                cloned_path = clone_ctx.__enter__()
                clone_managers.append(clone_ctx)
                path_to_analyze = cloned_path
            except RuntimeError as exc:
                console.print(f"[red]Clone failed:[/] {exc}")
                continue
            is_container = False
        else:
            is_container = is_docker_image(source)
            path_to_analyze = Path(source)

        source_summary = {
            "source_kind": (
                "git-url" if clone_ctx
                else "container" if is_container
                else "local-path"
            ),
            "status": "in_progress",
            "status_detail": None,
            "assets_discovered": 0,
            "branches_scanned": None,
            "last_generated_at": None,
            "errors": [],
        }
        source_outcomes[source] = source_summary

        if is_container:
            logging.info(f"Source '{source}' detected as a container image.")
            temp_dir = tempfile.mkdtemp(prefix="aibom_")
            extract_info = extract_app_from_docker(source, temp_dir)
            if "error" in extract_info:
                message = f"Error extracting from Docker image: {extract_info['error']}"
                logging.error(message)
                _record_analysis_error(
                    run_errors,
                    source_summary,
                    source,
                    message,
                    severity="fatal",
                )
                if temp_dir:
                    shutil.rmtree(temp_dir)
                continue
            path_to_analyze = Path(extract_info.get("extracted_to", temp_dir))

        if is_container:
            source_summary["source_path"] = "/app" if path_to_analyze.name == "app" else str(path_to_analyze)
            source_summary["source_name"] = str(source)
        else:
            source_summary["source_path"] = str(path_to_analyze.resolve())
            source_summary["source_name"] = Path(source).name or str(source)

        if not path_to_analyze.exists():
            message = f"Path or image '{source}' not found or could not be processed."
            logging.error(message)
            _record_analysis_error(
                run_errors,
                source_summary,
                source,
                message,
                severity="fatal",
            )
            if temp_dir:
                shutil.rmtree(temp_dir)
            continue

        # -----------------------------------------------------------------
        # v2 path: four-stage pipeline (scan → cross-ref → agentic → assemble)
        # -----------------------------------------------------------------
        if not legacy_mode:
            from .scan_pipeline import ScanPipeline

            scan_path = str(path_to_analyze)

            _scan_cache_hit = False
            if cache_dir:
                from .scan_cache import cache_key, load_cached, save_cached

                _ck = cache_key([scan_path])
                cached = load_cached(cache_dir, _ck)
                if cached:
                    _scan_cache_hit = True
                    console.print(f"[green]Cache hit[/] for {source} ({_ck[:12]}…)")
                    all_analysis_outputs[source] = cached
                    if temp_dir:
                        shutil.rmtree(temp_dir)
                    continue

            pipeline = ScanPipeline(
                scan_paths=[scan_path],
                output_format=output_format,
                output_file=str(output_file) if output_file else None,
                llm_config=llm_config,
                kb_path=str(db_path) if db_path else None,
                fail_on=fail_on_severity,
                min_severity=severity_filter,
                strict=strict,
                agentic_scope=agentic_scope,
                agentic_batch_size=agentic_batch_size,
                agentic_concurrency=agentic_concurrency,
                agentic_fast_model=agentic_fast_model,
            )
            with console.status(f"[cyan]Scanning {source} (v2 pipeline)"):
                result = pipeline.run()

            if llm_config and result.agentic_risk_flags:
                console.print(
                    f"  [magenta]Agentic enrichment added "
                    f"{len(result.agentic_risk_flags)} risk flags[/]"
                )

            if result.external_deps:
                escaping = [d for d in result.external_deps if d.escapes_root]
                if escaping:
                    lines = [
                        f"[yellow bold]{len(escaping)} dependency(ies) reference "
                        f"repos not included in this scan:[/]\n"
                    ]
                    for d in escaping[:10]:
                        label = d.name or d.url_or_path
                        lines.append(
                            f"  • [bold]{label}[/] ({d.dep_type}) "
                            f"from {Path(d.source_file).name}"
                        )
                    if len(escaping) > 10:
                        lines.append(f"  … and {len(escaping) - 10} more")
                    lines.append(
                        "\n[dim]Include these repos in your scan for "
                        "better cross-reference resolution.[/]"
                    )
                    console.print(Panel(
                        "\n".join(lines),
                        title="[bold]Missing Repositories[/]",
                        border_style="yellow",
                    ))

            if timing and result.timings:
                _print_timing_table(console, result)

            output_data: dict[str, Any] = {
                "_v2": True,
                "components": result.components,
                "relationships": result.relationships,
                "_agentic_risk_flags": result.agentic_risk_flags,
                "_agentic_candidate_count": result.agentic_candidate_count,
            }
            all_analysis_outputs[source] = output_data

            if cache_dir and not _scan_cache_hit:
                from .scan_cache import cache_key, save_cached

                _ck = cache_key([scan_path])
                _serializable = {
                    "_v2": True,
                    "components": [c.model_dump(mode="json") for c in result.components],
                    "relationships": [r.model_dump(mode="json") for r in result.relationships],
                    "_agentic_risk_flags": [
                        f.model_dump(mode="json") if hasattr(f, "model_dump") else str(f)
                        for f in result.agentic_risk_flags
                    ],
                    "_agentic_candidate_count": result.agentic_candidate_count,
                }
                save_cached(cache_dir, _ck, _serializable)

            source_summary["assets_discovered"] = len(result.components)
            source_summary["last_generated_at"] = _utcnow_iso()
            if source_summary["status"] == "in_progress":
                source_summary["status"] = "completed"

            if temp_dir:
                shutil.rmtree(temp_dir)
            continue

        # -----------------------------------------------------------------
        # Legacy path: KB-driven symbol matching (v1)
        # -----------------------------------------------------------------
        python_files = _find_python_files(path_to_analyze)
        if not python_files:
            logging.warning("No Python files found to analyze in this source.")
            source_summary["status"] = "skipped"
            source_summary["status_detail"] = "no_python_files"
            if temp_dir:
                shutil.rmtree(temp_dir)
            continue

        py_count = sum(1 for f in python_files if f.suffix == ".py")
        nb_count = sum(1 for f in python_files if f.suffix == ".ipynb")
        logging.info(f"Found {py_count} Python file(s) and {nb_count} notebook(s) to analyze...")

        analysis_results: List[CodeAnalysisResult] = []

        config_root = path_to_analyze if path_to_analyze.is_dir() else path_to_analyze.parent
        config_results = parse_project_configs(config_root)
        analysis_results.extend(config_results)

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task_id = progress.add_task(f"[cyan]Parsing {source}", total=len(python_files))
            for py_file in python_files:
                try:
                    if py_file.suffix == ".ipynb":
                        source_code = extract_code_from_notebook(py_file)
                        if not source_code.strip():
                            continue
                    else:
                        with open(py_file, "r", encoding="utf-8") as f:
                            source_code = f.read()
                    result = parse_source_code(str(py_file), source_code)
                    analysis_results.append(result)
                except Exception as e:
                    logging.warning(f"Could not parse {py_file}. Error: {e}")
                    _record_analysis_error(
                        run_errors,
                        source_summary,
                        source,
                        f"Could not parse {py_file}: {e}",
                        file_path=str(py_file),
                    )
                finally:
                    progress.advance(task_id)

        workflow_index = None
        try:
            with console.status(f"[green]Building workflow index for {source}"):
                workflow_index = build_workflow_index(python_files)
        except Exception as workflow_error:
            logging.debug(f"Failed to build workflow index: {workflow_error}")

        config_root = path_to_analyze if path_to_analyze.is_dir() else path_to_analyze.parent
        source_custom: Optional[CustomCatalogConfig] = explicit_config
        if source_custom is None:
            discovered = discover_custom_catalog(config_root)
            source_custom = load_custom_catalog(discovered) if discovered else None

        with CatalogDB(db_path) as connector:
            if source_custom and not source_custom.is_empty:
                connector.add_custom_entries(
                    [comp.to_catalog_dict() for comp in source_custom.components]
                )
                if source_custom.excludes:
                    connector.add_excludes(source_custom.excludes)
                n_comp = len(source_custom.components)
                n_base = len(source_custom.base_class_rules)
                n_excl = len(source_custom.excludes)
                n_rel = len(source_custom.custom_relationships)
                console.print(
                    f"[dim]Custom catalog: {n_comp} component(s), {n_base} base-class rule(s), "
                    f"{n_excl} exclude(s), {n_rel} custom relationship(s)[/]"
                )

            analysis_output = categorize_symbols(
                analysis_results,
                connector,
                llm_config,
                workflow_index,
                custom_config=source_custom,
            )

        if is_container and temp_dir:
            analysis_output = _convert_paths_in_output(analysis_output, temp_dir)

        all_analysis_outputs[source] = analysis_output
        categorized_components = getattr(analysis_output, "components", analysis_output)
        total_components = sum(len(items or []) for items in categorized_components.values())
        source_summary["assets_discovered"] = total_components
        source_summary["last_generated_at"] = _utcnow_iso()
        if source_summary["status"] == "in_progress":
            source_summary["status"] = "completed"

        if temp_dir:
            shutil.rmtree(temp_dir)

    run_metadata["completed_at"] = _utcnow_iso()
    run_metadata["error_count"] = len(run_errors)
    run_metadata["sources_analyzed"] = len(source_outcomes)
    sources_with_errors = sum(
        1
        for info in source_outcomes.values()
        if info.get("status") in {"completed_with_errors", "failed"}
    )
    run_metadata["sources_with_errors"] = sources_with_errors
    if any(info.get("status") == "failed" for info in source_outcomes.values()):
        run_metadata["status"] = "failed"
    elif run_errors:
        run_metadata["status"] = "completed_with_errors"
    else:
        run_metadata["status"] = "completed"

    # Cross-repo coordination (agentic, multi-source only)
    v2_outputs = {
        k: v for k, v in all_analysis_outputs.items()
        if isinstance(v, dict) and v.get("_v2")
    }
    if llm_config and len(v2_outputs) > 1:
        try:
            from .agentic.agent import run_cross_repo_coordination

            console.print(
                f"  [cyan]Cross-repo coordination across "
                f"{len(v2_outputs)} repos…[/]"
            )
            xrepo_rels, xrepo_flags = run_cross_repo_coordination(
                model_string=llm_config["model"],
                per_repo_results=v2_outputs,
                llm_config=llm_config,
            )
            if xrepo_rels or xrepo_flags:
                first_key = next(iter(v2_outputs))
                first = all_analysis_outputs[first_key]
                first.setdefault("relationships", []).extend(xrepo_rels)
                first.setdefault("_agentic_risk_flags", []).extend(xrepo_flags)
                console.print(
                    f"  [magenta]Cross-repo: {len(xrepo_rels)} relationships, "
                    f"{len(xrepo_flags)} risk flags[/]"
                )
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Cross-repo coordination failed: %s", exc)

    # Phase 4: Generate the report
    report_data = None
    reporter = get_reporter(output_format)

    has_v2_output = any(
        isinstance(o, dict) and o.get("_v2")
        for o in all_analysis_outputs.values()
    )

    if has_v2_output or (reporter and output_format not in ("json", "plaintext")):
        from .models import (
            AIComponent as V2Component,
            AIComponentType,
            ComponentRelationship as V2Relationship,
            ScanResult,
            SourceResult,
        )
        from .risk import RiskScorer

        if has_v2_output:
            v2_sources = []
            for source_path, output in all_analysis_outputs.items():
                if isinstance(output, dict) and output.get("_v2"):
                    v2_sources.append(SourceResult(
                        path=source_path,
                        components=output["components"],
                        relationships=output["relationships"],
                    ))
                else:
                    legacy_sr = _build_scan_result(
                        {source_path: output}, run_metadata, [],
                    )
                    v2_sources.extend(legacy_sr.sources)
            scan_result = ScanResult(
                metadata=run_metadata,
                sources=v2_sources,
                errors=[e.get("message", str(e)) for e in run_errors],
            )
        else:
            scan_result = _build_scan_result(
                all_analysis_outputs, run_metadata, run_errors,
            )

        scorer = RiskScorer()
        scan_result.risk = scorer.score(scan_result)

        for output in all_analysis_outputs.values():
            if isinstance(output, dict):
                for rf in output.get("_agentic_risk_flags", []):
                    scan_result.risk.add_flag(rf)

        if not reporter:
            reporter = get_reporter(output_format)

        if reporter:
            if validate:
                errors = reporter.validate(scan_result)
                if errors:
                    for err in errors:
                        console.print(f"[yellow]Validation: {err}[/]")
                else:
                    console.print("[green]Validation passed.[/]")

            if output_file:
                import io
                buf = io.StringIO()
                reporter.render(scan_result, buf)
                output_file.write_text(buf.getvalue(), encoding="utf-8")
                console.print(f"[green]Report written to {output_file}[/]")
            else:
                import sys
                reporter.render(scan_result, sys.stdout)
        elif output_format == "json":
            import io
            json_rep = get_reporter("json")
            if json_rep and output_file:
                buf = io.StringIO()
                json_rep.render(scan_result, buf)
                output_file.write_text(buf.getvalue(), encoding="utf-8")
                console.print(f"[green]Report written to {output_file}[/]")
        elif output_format == "plaintext":
            plaintext_rep = get_reporter("plaintext")
            if plaintext_rep and output_file:
                import io
                buf = io.StringIO()
                plaintext_rep.render(scan_result, buf)
                output_file.write_text(buf.getvalue(), encoding="utf-8")
                console.print(f"[green]Report written to {output_file}[/]")

        if fail_on_severity and scorer.should_fail(scan_result.risk, fail_on_severity):
            console.print(
                f"[bold red]Risk threshold exceeded: {scan_result.risk.severity.value} "
                f">= {fail_on_severity.value}[/]"
            )
            raise typer.Exit(code=2)

    elif output_format == "json":
        report_data = _generate_json_report(
            all_analysis_outputs,
            output_file,
            metadata=run_metadata,
            source_summaries=source_outcomes,
            run_errors=run_errors,
        )
        if post_url:
            try:
                submission_payload = _build_submission_payload(report_data, source_outcomes)
                logging.info("Sending report")
                post_report_with_retries(
                    post_url,
                    submission_payload,
                    api_key=ai_defense_api_key,
                    verify_tls=post_verify_tls,
                    timeout_seconds=post_timeout,
                )
                logging.info("Report uploaded to %s", post_url)
            except Exception as exc:  # noqa: BLE001
                logging.error("Failed to POST report: %s", exc)
                raise typer.Exit(code=1)
    elif output_format == "api":
        logging.info("--- Starting API Server ---")
        component_map = {
            source: getattr(output, "components", output) for source, output in all_analysis_outputs.items()
        }
        start_api_server(component_map)
    else:  # plaintext
        _generate_plaintext_report(all_analysis_outputs, output_file)

    for cm in clone_managers:
        cm.__exit__(None, None, None)

    total_agentic = sum(
        v.get("_agentic_candidate_count", 0)
        for v in all_analysis_outputs.values()
        if isinstance(v, dict)
    )
    if total_agentic > 0 and not llm_config:
        console.print()
        console.print(
            Panel.fit(
                f"[yellow bold]{total_agentic} detection(s) need agentic reasoning[/]\n\n"
                "These are ambiguous patterns (e.g., model names in IaC values,\n"
                ".fit() calls without ML imports, generic Agent/Tool usage)\n"
                "that require LLM reasoning to confirm or discard.\n\n"
                "[green]Re-run with --llm-model <model> to resolve them.[/]\n"
                "[dim]Use --strict to suppress these from the report.[/]",
                title="[bold]Agentic Reasoning Recommended[/]",
                border_style="yellow",
            )
        )

    if show_summary:
        _display_analysis_summary(all_analysis_outputs)

kb_app = typer.Typer(help="Manage the AIBOM knowledge base.", no_args_is_help=True)
app.add_typer(kb_app, name="kb")


# ---------------------------------------------------------------------------
# cisco-aibom cache  — manage scan result cache
# ---------------------------------------------------------------------------

cache_app = typer.Typer(help="Manage the scan result cache.", no_args_is_help=True)
app.add_typer(cache_app, name="cache")


@cache_app.command("clear")
def cache_clear(
    cache_dir: Path = typer.Option(
        Path.home() / ".aibom" / "cache",
        "--cache-dir",
        help="Directory where cached scan results are stored.",
    ),
) -> None:
    """Remove all cached scan results."""
    from .scan_cache import clear_cache as _clear

    removed = _clear(cache_dir)
    console.print(f"[green]Removed {removed} cached scan result(s) from {cache_dir}[/]")


@cache_app.command("list")
def cache_list(
    cache_dir: Path = typer.Option(
        Path.home() / ".aibom" / "cache",
        "--cache-dir",
        help="Directory where cached scan results are stored.",
    ),
) -> None:
    """List all cached scan results."""
    from rich.table import Table

    from .scan_cache import cache_info

    entries = cache_info(cache_dir)
    if not entries:
        console.print("[dim]No cached scan results found.[/]")
        return

    table = Table(title=f"Scan Cache ({cache_dir})")
    table.add_column("Key")
    table.add_column("Cached At")
    table.add_column("Size", justify="right")
    for e in entries:
        table.add_row(e["key"][:16] + "…", e["cached_at"], f"{e['size_kb']} KB")
    console.print(table)


# ---------------------------------------------------------------------------
# cisco-aibom plugin  — discover and manage plugins
# ---------------------------------------------------------------------------

plugin_app = typer.Typer(help="Discover and manage AIBOM plugins.", no_args_is_help=True)
app.add_typer(plugin_app, name="plugin")


@plugin_app.command("list")
def plugin_list() -> None:
    """List all discovered plugins (entry_points, MCP servers, manifests)."""
    from rich.table import Table

    from .plugins import list_plugins

    plugins = list_plugins()

    for category, items in plugins.items():
        if not items:
            continue
        table = Table(title=f"{category.replace('_', ' ').title()}")
        if items:
            for col in items[0]:
                table.add_column(col.replace("_", " ").title())
            for item in items:
                table.add_row(*item.values())
        console.print(table)

    total = sum(len(v) for v in plugins.values())
    if total == 0:
        console.print("[dim]No plugins discovered.[/]")
        console.print(
            "\n[dim]To create a scanner plugin, add to your pyproject.toml:[/]\n"
            '  [project.entry-points."aibom.scanners"]\n'
            '  my_scanner = "my_package:MyScannerClass"\n'
        )


@kb_app.command("download")
def kb_download(
    version: Optional[str] = typer.Option(None, "--version", "-v", help="Specific KB version to download (latest if omitted)."),
    url: Optional[str] = typer.Option(None, "--url", help="Override the manifest URL."),
) -> None:
    """Download the knowledge base from Cisco's public repository."""
    mgr = KBManager()
    try:
        path = mgr.download(version=version, url=url)
        console.print(f"[green]KB downloaded to {path}[/]")
    except KBError as exc:
        console.print(f"[bold red]Download failed:[/] {exc}")
        raise typer.Exit(code=1)


@kb_app.command("check")
def kb_check() -> None:
    """Check if a newer knowledge base version is available."""
    mgr = KBManager()
    try:
        info = mgr.check()
        console.print(f"Current version: {info['current_version']}")
        console.print(f"Latest version:  {info['latest_version']}")
        if info["update_available"]:
            console.print("[yellow]Update available![/] Run: cisco-aibom kb download")
        else:
            console.print("[green]You have the latest version.[/]")
    except KBError as exc:
        console.print(f"[bold red]Check failed:[/] {exc}")
        raise typer.Exit(code=1)


@kb_app.command("info")
def kb_info() -> None:
    """Display information about the locally installed knowledge base."""
    mgr = KBManager()
    try:
        info = mgr.info()
        table = Table(title="Knowledge Base Info", box=box.SIMPLE_HEAVY)
        table.add_column("Property", style="bold")
        table.add_column("Value")
        for key, value in info.items():
            table.add_row(key, str(value))
        console.print(table)
    except KBError as exc:
        console.print(f"[bold red]Info failed:[/] {exc}")
        raise typer.Exit(code=1)


@kb_app.command("verify")
def kb_verify() -> None:
    """Verify the integrity of the locally installed knowledge base."""
    mgr = KBManager()
    if mgr.verify():
        console.print("[green]Knowledge base integrity verified.[/]")
    else:
        console.print("[bold red]Knowledge base integrity check failed.[/]")
        raise typer.Exit(code=1)


@kb_app.command("request")
def kb_request(
    sdk: str = typer.Option(..., "--sdk", help="SDK name (e.g., langchain, openai)."),
    version: str = typer.Option(..., "--version", "-v", help="SDK version to request KB build for."),
    language: str = typer.Option("python", "--language", "-l", help="Programming language."),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", envvar="CISCO_AI_DEFENSE_API_KEY",
        help="Cisco AI Defense API key.",
    ),
    api_base: Optional[str] = typer.Option(
        None, "--api-base", envvar="CISCO_AI_DEFENSE_API_BASE",
        help="Cisco AI Defense API base URL.",
    ),
) -> None:
    """Request a knowledge base build for a specific SDK version."""
    mgr = KBManager()
    try:
        result = mgr.request_build(sdk=sdk, version=version, language=language, api_key=api_key, api_base=api_base)
        console.print(f"[green]Request submitted:[/] {result.get('request_id', 'unknown')}")
        console.print(f"Status: {result.get('status', 'unknown')}")
    except KBError as exc:
        console.print(f"[bold red]Request failed:[/] {exc}")
        raise typer.Exit(code=1)


@kb_app.command("request-status")
def kb_request_status(
    request_id: str = typer.Argument(..., help="Request ID to check."),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", envvar="CISCO_AI_DEFENSE_API_KEY",
    ),
    api_base: Optional[str] = typer.Option(
        None, "--api-base", envvar="CISCO_AI_DEFENSE_API_BASE",
    ),
) -> None:
    """Check the status of a KB build request."""
    mgr = KBManager()
    try:
        result = mgr.request_status(request_id, api_key=api_key, api_base=api_base)
        for key, value in result.items():
            console.print(f"{key}: {value}")
    except KBError as exc:
        console.print(f"[bold red]Status check failed:[/] {exc}")
        raise typer.Exit(code=1)


@kb_app.command("list-requests")
def kb_list_requests(
    api_key: Optional[str] = typer.Option(
        None, "--api-key", envvar="CISCO_AI_DEFENSE_API_KEY",
    ),
    api_base: Optional[str] = typer.Option(
        None, "--api-base", envvar="CISCO_AI_DEFENSE_API_BASE",
    ),
) -> None:
    """List all pending KB build requests."""
    mgr = KBManager()
    try:
        requests = mgr.list_requests(api_key=api_key, api_base=api_base)
        if not requests:
            console.print("No pending requests.")
            return
        table = Table(title="KB Build Requests", box=box.SIMPLE_HEAVY)
        table.add_column("Request ID")
        table.add_column("SDK")
        table.add_column("Version")
        table.add_column("Language")
        table.add_column("Status")
        for req in requests:
            table.add_row(
                req.get("request_id", ""),
                req.get("sdk", ""),
                req.get("version", ""),
                req.get("language", ""),
                req.get("status", ""),
            )
        console.print(table)
    except KBError as exc:
        console.print(f"[bold red]List failed:[/] {exc}")
        raise typer.Exit(code=1)


def cli_entry_point() -> None:
    """Entry point for console_scripts."""
    import sys
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 5000))
    app()


if __name__ == "__main__":
    cli_entry_point()
