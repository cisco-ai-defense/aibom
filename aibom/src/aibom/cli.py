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
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from .report_sender import post_report_with_retries
from .utils.version import resolve_package_version
from .categorizer import categorize_symbols
from .container_utils import extract_app_from_docker, is_docker_image
from .cst_parser import parse_source_code
from .catalog_db import CatalogDB
from .db_loader import ensure_local_database
from .structures import CodeAnalysisResult
from .ui_handler import start_ui_server
from .workflow_analyzer import build_workflow_index, workflow_identifier

console = Console()

app = typer.Typer(
    help="Generate an AI BOM from Python source code.",
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
            "[bold red]No subcommand provided.[/] Please use [green]analyze[/] or [green]report[/]."
        )
        raise typer.Exit(code=1)


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


def _display_analysis_summary(all_analysis_outputs: Dict[str, Any], max_examples: int = 3) -> None:
    for source, output in all_analysis_outputs.items():
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
    """Finds all .py files in a given path (file or directory)."""
    if path.is_file() and path.suffix == ".py":
        return [path]
    if path.is_dir():
        return list(path.rglob("*.py"))
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
        help="Output format (json, plaintext, ui)",
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
        help="LLM model name for semantic model extraction (e.g., gpt-3.5-turbo)",
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
    if output_format not in ["plaintext", "json", "ui"]:
        logging.error(f"Invalid output format '{output_format}'. Must be 'plaintext', 'json', or 'ui'.")
        raise typer.Exit(code=1)
    
    # Validate LLM configuration if model extraction is requested
    llm_config = None
    if llm_model:
        if not llm_api_base:
            logging.error("--llm-api-base is required when --llm-model is specified.")
            raise typer.Exit(code=1)
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
    connector: Optional[CatalogDB] = None
    db_path: Optional[Path] = None
    try:
        db_path = ensure_local_database(console=console)
        connector = CatalogDB(db_path)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Knowledge base error:[/] {exc}")
        raise typer.Exit(code=1)
    try:
        for source in sources_to_process:
            console.print(Panel.fit(f"Analyzing Source: {source}", style="bold cyan"))
            temp_dir = None
            is_container = is_docker_image(source)
            path_to_analyze = Path(source)
            source_summary = {
                "source_kind": "container" if is_container else "local-path",
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

            python_files = _find_python_files(path_to_analyze)
            if not python_files:
                logging.warning("No Python files found to analyze in this source.")
                source_summary["status"] = "skipped"
                source_summary["status_detail"] = "no_python_files"
                if temp_dir:
                    shutil.rmtree(temp_dir)
                continue

            logging.info(f"Found {len(python_files)} Python file(s) to analyze...")

            analysis_results: List[CodeAnalysisResult] = []
            with Progress(
                SpinnerColumn(),
                "[progress.description]{task.description}",
                TimeElapsedColumn(),
                transient=True,
            ) as progress:
                task_id = progress.add_task(f"[cyan]Parsing {source}", total=len(python_files))
                for py_file in python_files:
                    try:
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

            analysis_output = categorize_symbols(
                analysis_results, connector, llm_config, workflow_index
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
    finally:
        if connector:
            connector.close()

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

    # Phase 4: Generate the report
    report_data = None
    if output_format == "json":
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
    elif output_format == "ui":
        logging.info("--- Starting UI Server ---")
        component_map = {
            source: getattr(output, "components", output) for source, output in all_analysis_outputs.items()
        }
        start_ui_server(component_map)
    else:  # plaintext
        _generate_plaintext_report(all_analysis_outputs, output_file)

    if show_summary:
        _display_analysis_summary(all_analysis_outputs)

def cli_entry_point() -> None:
    """Entry point for console_scripts."""
    app()


if __name__ == "__main__":
    cli_entry_point()
