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

import hashlib
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
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from .report_sender import post_report_with_retries
from .utils.version import resolve_package_version
from .scanners.container_extractor import extract_source_from_image, is_container_image, VALID_TIERS, validate_tier
from .cache_paths import (
    cache_dir as resolve_cache_type_dir,
    cache_read_dirs,
    cache_types,
    resolve_cache_root,
)
from .llm_factory import ensure_llm_runtime_available
from .custom_catalog import (
    CustomCatalogConfig,
    load_custom_catalog,
)
from .api_handler import start_api_server
from .reporters import get_reporter, reporter_registry
from .models.enums import Severity as SeverityEnum
from .kb.manager import KBManager, KBError

_LOGGER = logging.getLogger(__name__)

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


def _should_render_progress(progress_enabled: Optional[bool]) -> bool:
    """Choose whether to show live progress for the current terminal."""
    if progress_enabled is not None:
        return progress_enabled
    return console.is_terminal and not os.environ.get("CI")


def _run_pipeline_with_progress(source: str, pipeline: "ScanPipeline", progress_enabled: Optional[bool]) -> "PipelineResult":  # type: ignore[name-defined]
    """Run a scan pipeline with either a Rich progress display or a spinner."""
    if not _should_render_progress(progress_enabled):
        with console.status(f"[cyan]Scanning {source}"):
            return pipeline.run()

    stage_labels = {
        "scan": "Stage 1/4: deterministic scanners",
        "cross_ref": "Stage 2/4: cross-reference resolution",
        "agentic": "Stage 3/4: agentic enrichment",
        "assemble": "Stage 4/4: final assembly",
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress_ui:
        stage_task = progress_ui.add_task(f"{source}: preparing scan", total=4)
        cache_task = progress_ui.add_task(
            f"{source}: file cache prep",
            total=1,
            completed=0,
            visible=False,
        )
        scanner_task = progress_ui.add_task(
            f"{source}: deterministic scanners",
            total=1,
            completed=0,
            visible=False,
        )

        def _on_progress(event: dict[str, Any]) -> None:
            event_name = str(event.get("event", ""))
            if event_name == "stage_started":
                stage = str(event.get("stage", ""))
                progress_ui.update(
                    stage_task,
                    description=f"{source}: {stage_labels.get(stage, stage)}",
                )
                return
            if event_name == "stage_completed":
                progress_ui.advance(stage_task, 1)
                return
            if event_name == "file_cache_prep_started":
                total = max(int(event.get("files_total", 0) or 0), 1)
                progress_ui.update(
                    cache_task,
                    total=total,
                    completed=0,
                    visible=True,
                    description=f"{source}: file cache prep",
                )
                return
            if event_name == "file_cache_prep_completed":
                total = max(int(event.get("files_total", 0) or 0), 1)
                warmed = min(int(event.get("files_warmed", 0) or 0), total)
                progress_ui.update(
                    cache_task,
                    total=total,
                    completed=warmed,
                    visible=True,
                    description=f"{source}: file cache prep",
                )
                return
            if event_name == "scanners_discovered":
                total = max(int(event.get("total", 0) or 0), 1)
                progress_ui.update(
                    scanner_task,
                    total=total,
                    completed=0,
                    visible=True,
                    description=f"{source}: deterministic scanners",
                )
                return
            if event_name == "scanner_completed":
                total = max(int(event.get("total", 0) or 0), 1)
                completed = min(int(event.get("completed", 0) or 0), total)
                scanner = str(event.get("scanner", "scanner"))
                progress_ui.update(
                    scanner_task,
                    total=total,
                    completed=completed,
                    visible=True,
                    description=f"{source}: scanner {completed}/{total} ({scanner})",
                )

        pipeline.progress_callback = _on_progress
        return pipeline.run()


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


def _file_cache_fingerprint(path: Optional[Path]) -> Optional[dict[str, Any]]:
    """Return a stable fingerprint for an explicit file-based analysis input."""
    if path is None:
        return None
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        resolved = path.expanduser()
    info: dict[str, Any] = {"path": str(resolved)}
    try:
        info["sha256"] = hashlib.sha256(resolved.read_bytes()).hexdigest()[:16]
    except OSError:
        info["exists"] = resolved.exists()
    return info


def _scan_cache_settings(
    *,
    strict: bool,
    min_severity: SeverityEnum,
    llm_config: Optional[Dict[str, Any]],
    agentic_scope: str,
    agentic_batch_size: int,
    agentic_concurrency: int,
    agentic_fast_model: Optional[str],
    agentic_timeout: int,
    include_code_snippets: bool,
    container_tier: str,
    custom_catalog: Optional[Path],
) -> Dict[str, Any]:
    """Build a stable settings payload for persistent scan-cache keys."""
    safe_llm = None
    if llm_config:
        safe_llm = {
            "model": llm_config.get("model"),
            "provider": llm_config.get("provider"),
            "api_base": llm_config.get("api_base"),
            "api_version": llm_config.get("api_version"),
        }
    return {
        "strict": strict,
        "min_severity": min_severity.value,
        "llm_config": safe_llm,
        "agentic_scope": agentic_scope,
        "agentic_batch_size": agentic_batch_size,
        "agentic_concurrency": agentic_concurrency,
        "agentic_fast_model": agentic_fast_model,
        "agentic_timeout": agentic_timeout,
        "include_code_snippets": include_code_snippets,
        "container_tier": container_tier,
        "custom_catalog": _file_cache_fingerprint(custom_catalog),
    }


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        envvar="AIBOM_LOG_LEVEL",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    ),
) -> None:
    """Generate an AI BOM from source code."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        console.print(f"[red]Invalid log level:[/] {log_level}")
        raise typer.Exit(code=1)
    logging.basicConfig(level=numeric_level, format="%(levelname)s: %(message)s")

    if ctx.invoked_subcommand is None:
        console.print(
            "[bold red]No subcommand provided.[/] Please use [green]analyze[/], "
            "[green]report[/], [green]watch[/], [green]benchmark[/], [green]kb[/], "
            "[green]cache[/], [green]plugin[/], or [green]diff[/]."
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


def _display_v2_summary(source: str, components: list, relationships: list) -> None:
    """Rich summary for v2 detector output."""
    from collections import Counter
    from .models import AIComponent as _V2Comp
    components = [
        _V2Comp.model_validate(c) if isinstance(c, dict) else c
        for c in components
    ]
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
    source_outcomes: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Wrap the generated report in the API submission envelope."""
    analysis = report.get("aibom_analysis", {})
    metadata = analysis.get("metadata", {})
    if not source_outcomes:
        source_outcomes = _source_outcomes_from_report(report)

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


def _source_outcomes_from_report(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Reconstruct per-source submission metadata from an on-disk JSON report."""
    analysis = report.get("aibom_analysis", {})
    sources = analysis.get("sources", {})
    outcomes: Dict[str, Dict[str, Any]] = {}
    for source_key, entry in sources.items():
        if not isinstance(entry, dict):
            continue
        summary = entry.get("summary", {}) if isinstance(entry.get("summary"), dict) else {}
        source_path = str(entry.get("source_path") or source_key)
        outcomes[source_path] = {
            "source_name": str(entry.get("source_name") or source_key),
            "source_path": source_path,
            "source_kind": summary.get("source_kind"),
            "status": summary.get("status"),
            "last_generated_at": summary.get("last_generated_at"),
            "assets_discovered": summary.get("assets_discovered"),
        }
    return outcomes


def _load_report_json_dict(report_file: Path) -> Dict[str, Any]:
    """Read a report file and ensure it is valid JSON object content."""
    try:
        data = json.loads(report_file.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Failed to read report: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Report JSON must contain an object at the top level")
    return data


def _canonicalize_report_for_upload(report_file: Path) -> tuple[Dict[str, Any], bool]:
    """Validate a report file and rebuild the canonical upload payload shape."""
    from pydantic import ValidationError

    from .diff import load_scan_result_json
    from .reporters.json_reporter import _aibom_payload

    raw_report = _load_report_json_dict(report_file)
    analysis = raw_report.get("aibom_analysis")
    if not isinstance(analysis, dict):
        raise ValueError("Report does not contain an 'aibom_analysis' key")

    legacy_schema = not bool(analysis.get("metadata", {}).get("report_schema_version"))
    try:
        scan_result = load_scan_result_json(report_file)
    except ValidationError as exc:
        raise ValueError(f"Invalid report structure: {exc}") from exc

    return _aibom_payload(scan_result), legacy_schema


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


def _show_report_impl(report_file: Path, *, raw: bool = False) -> None:
    data = _load_report_json_dict(report_file)
    if raw:
        console.print(Syntax(json.dumps(data, indent=2), "json", theme="monokai"))
    _render_json_report_console(data)


@app.command("report")
def report_command(
    action_or_file: str = typer.Argument(
        ...,
        help="Either a report file path, or one of: show, upload.",
    ),
    report_file: Optional[Path] = typer.Argument(
        None,
        readable=True,
        dir_okay=False,
        resolve_path=True,
        help="Path to the JSON report file when using 'show' or 'upload'.",
    ),
    raw: bool = typer.Option(False, "--raw-json", help="Display the raw JSON using syntax highlighting before the summary."),
    report_format: str = typer.Option(
        "json",
        "--format",
        help="Upload format. Only 'json' is currently supported.",
    ),
    post_url: Optional[str] = typer.Option(
        None,
        "--post-url",
        help="HTTP endpoint to POST the JSON report to (can also be set via AIBOM_POST_URL).",
        envvar="AIBOM_POST_URL",
    ),
    ai_defense_api_key: Optional[str] = typer.Option(
        None,
        "--ai-defense-api-key",
        help="API key sent when POSTing the report.",
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
) -> None:
    """Show or upload a previously generated JSON report."""
    action = action_or_file.lower()
    if action in {"show", "upload"}:
        target_report = report_file
        if target_report is None:
            console.print(f"[red]Missing report file for 'report {action}'.[/]")
            raise typer.Exit(code=1)
    else:
        if report_file is not None:
            console.print(
                "[red]Unexpected extra argument.[/] "
                "Use 'cisco-aibom report <report.json>' or "
                "'cisco-aibom report show|upload <report.json>'."
            )
            raise typer.Exit(code=1)
        action = "show"
        target_report = Path(action_or_file).expanduser()

    if not target_report.exists() or not target_report.is_file():
        console.print(f"[red]Report file not found:[/] {target_report}")
        raise typer.Exit(code=1)

    if action == "show":
        try:
            _show_report_impl(target_report, raw=raw)
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=1)
        return

    if report_format.lower() != "json":
        console.print(
            f"[red]Unsupported report upload format:[/] {report_format}. "
            "Only 'json' is currently supported."
        )
        raise typer.Exit(code=1)
    if not post_url:
        console.print("[red]--post-url is required for 'report upload'.[/]")
        raise typer.Exit(code=1)

    try:
        canonical_report, legacy_schema = _canonicalize_report_for_upload(target_report)
        if legacy_schema:
            console.print(
                "[yellow]Warning:[/] Report uses a deprecated schema without "
                "`report_schema_version`; synthesizing the current schema for upload."
            )
        submission_payload = _build_submission_payload(canonical_report)
        post_report_with_retries(
            post_url,
            submission_payload,
            api_key=ai_defense_api_key,
            verify_tls=post_verify_tls,
            timeout_seconds=post_timeout,
        )
        console.print(f"[green]Report uploaded to {post_url}[/]")
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Failed to POST report:[/] {exc}")
        raise typer.Exit(code=1)


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
    llm_model: Optional[str] = typer.Option(
        None,
        "--llm-model",
        envvar="AIBOM_LLM_MODEL",
        help=(
            "LLM model name (e.g. gpt-5.4, us.anthropic.claude-sonnet-4-20250514-v1:0).  "
            "Required: the LLM agent classifies every scanner candidate for "
            "accurate results (requires 'cisco-aibom[agentic]').  "
            "Set via --llm-model or AIBOM_LLM_MODEL env var.  "
            "The legacy 'provider/model' prefix is still accepted."
        ),
    ),
    llm_provider: Optional[str] = typer.Option(
        None,
        "--llm-provider",
        envvar="AIBOM_LLM_PROVIDER",
        help=(
            "LLM provider name for LangChain (e.g. bedrock, openai, anthropic, "
            "azure_openai, google_genai, ollama).  If not set, inferred from "
            "the model name or a 'provider/' prefix in --llm-model."
        ),
    ),
    llm_api_key: Optional[str] = typer.Option(
        None,
        "--llm-api-key",
        envvar="AIBOM_LLM_API_KEY",
        help="LLM API key. May be optional for local LLM or AWS Bedrock.",
    ),
    llm_api_base: Optional[str] = typer.Option(
        None,
        "--llm-api-base",
        envvar="AIBOM_LLM_API_BASE",
        help="LLM API base URL.",
    ),
    llm_api_version: Optional[str] = typer.Option(
        None,
        "--llm-api-version",
        envvar="AIBOM_LLM_API_VERSION",
        help="LLM API version (for Azure OpenAI). May be optional for other providers.",
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
    policy: Optional[str] = typer.Option(
        None,
        "--policy",
        help="Path to a YAML policy file. When set, violations are printed and the process exits with code 1 if the policy does not pass.",
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
    progress: Optional[bool] = typer.Option(
        None,
        "--progress/--no-progress",
        help=(
            "Show live per-stage and per-scanner progress during analysis. "
            "Defaults to auto when attached to an interactive terminal."
        ),
    ),
    agentic_scope: str = typer.Option(
        "all",
        "--agentic-scope",
        hidden=True,
        help="Deprecated: all components are now sent to the agent. Kept for backward compatibility.",
    ),
    agentic_batch_size: int = typer.Option(
        5,
        "--agentic-batch-size",
        help="Components per agentic LLM invocation (1–50, default 5).",
        min=1,
        max=50,
    ),
    agentic_concurrency: int = typer.Option(
        1,
        "--agentic-concurrency",
        help="Max parallel agentic LLM batches (1–8, default 1).",
        min=1,
        max=8,
    ),
    agentic_fast_model: Optional[str] = typer.Option(
        None,
        "--agentic-fast-model",
        help=(
            "Cheaper/faster LLM for simple confirmations (model lookups, "
            "dependency checks). Falls back to --llm-model if not set."
        ),
    ),
    agentic_timeout: int = typer.Option(
        120,
        "--agentic-timeout",
        help="Wall-clock timeout (seconds) per agentic LLM batch (default 120).",
    ),
    include_code_snippets: bool = typer.Option(
        False,
        "--include-code-snippets/--no-code-snippets",
        help=(
            "Include raw code snippets in per-finding decision annotations. "
            "Disabled by default to limit report size and source exposure."
        ),
    ),
    container_tier: str = typer.Option(
        "auto",
        "--container-extraction-tier",
        help=(
            "Force a specific container extraction tier instead of auto-detection. "
            f"Choices: {', '.join(VALID_TIERS)}."
        ),
        callback=lambda v: validate_tier(v),
    ),
    cache_dir: Optional[Path] = typer.Option(
        None,
        "--cache-dir",
        help=(
            "Cache root directory. Scan results are stored under <cache-dir>/scan "
            "and agentic cache under <cache-dir>/agentic. "
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
    skip_unchanged: bool = typer.Option(
        False,
        "--skip-unchanged",
        help=(
            "Skip scanning git repos whose HEAD has not changed since "
            "the last cached scan. Cache is written under "
            "~/.aibom/cache/org by default and still reads legacy org-cache "
            "locations for compatibility."
        ),
    ),
    compliance: Optional[str] = typer.Option(
        None,
        "--compliance",
        help=(
            "Advisory compliance evaluation after scan: eu-ai-act, "
            "owasp-agentic, nist-ai-rmf, or all (does not change exit code)."
        ),
    ),
):
    """Analyzes a Python codebase to generate an AI BOM."""
    # Validate output format
    if output_format not in _VALID_OUTPUT_FORMATS:
        valid = ", ".join(sorted(_VALID_OUTPUT_FORMATS))
        console.print(f"[red]Invalid output format[/] '{output_format}'. Must be one of: {valid}")
        raise typer.Exit(code=1)

    agentic_scope = "all"

    if compliance is not None:
        allowed_cf = frozenset({"eu-ai-act", "owasp-agentic", "nist-ai-rmf", "all"})
        if compliance.strip().lower() not in allowed_cf:
            logging.error(
                "Invalid --compliance %r. Must be one of: %s",
                compliance,
                ", ".join(sorted(allowed_cf)),
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
            "provider": llm_provider,
            "api_key": llm_api_key,
            "api_base": llm_api_base,
            "api_version": llm_api_version,
        }
        try:
            ensure_llm_runtime_available(
                llm_model,
                provider=llm_provider,
            )
        except ImportError as exc:
            console.print(
                f"[bold red]Error:[/] {escape(str(exc))}",
                highlight=False,
            )
            raise typer.Exit(code=1)
    else:
        console.print(
            "[bold red]Error:[/] --llm-model (or AIBOM_LLM_MODEL env var) is required.\n"
            "The LLM agent classifies every scanner candidate for accurate results.\n"
            "Example: cisco-aibom analyze --llm-model gpt-5.4 ./my-repo",
            highlight=False,
        )
        raise typer.Exit(code=1)
    
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
        if llm_provider:
            triage_llm_cfg["provider"] = llm_provider
        if llm_api_base:
            triage_llm_cfg["api_base"] = llm_api_base
        if llm_api_key:
            triage_llm_cfg["api_key"] = llm_api_key
        if llm_api_version:
            triage_llm_cfg["api_version"] = llm_api_version

        triager = RepoTriager(llm_config=triage_llm_cfg)
        triage_results = triager.triage_repos(sources_to_process)

        deep = [t.repo_path for t in triage_results if t.decision == "deep-scan"]
        clone = [t.repo_path for t in triage_results if t.decision == "needs-clone"]
        skipped = [t for t in triage_results if t.decision == "skip"]

        if skipped:
            for t in skipped:
                _LOGGER.info(
                    "Triage would skip %s — %s (%s); keeping because user-provided",
                    t.repo_path, t.reason, t.method,
                )
            clone.extend(t.repo_path for t in skipped)

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
    explicit_config: Optional[CustomCatalogConfig] = None
    if custom_catalog:
        explicit_config = load_custom_catalog(Path(custom_catalog))
    cache_root = resolve_cache_root(cache_dir)
    scan_cache_dir = resolve_cache_type_dir("scan", cache_root)
    scan_cache_read_dirs = [p for p in cache_read_dirs("scan", cache_root) if p != scan_cache_dir]
    scan_cache_settings = _scan_cache_settings(
        strict=strict,
        min_severity=severity_filter,
        llm_config=llm_config,
        agentic_scope=agentic_scope,
        agentic_batch_size=agentic_batch_size,
        agentic_concurrency=agentic_concurrency,
        agentic_fast_model=agentic_fast_model,
        agentic_timeout=agentic_timeout,
        include_code_snippets=include_code_snippets,
        container_tier=container_tier,
        custom_catalog=custom_catalog,
    )
    agentic_cache_dir = resolve_cache_type_dir("agentic", cache_root)

    clone_managers: list[Any] = []

    for source in sources_to_process:
        console.print(Panel.fit(f"Analyzing Source: {source}", style="bold cyan"))
        temp_dir = None
        clone_ctx = None

        from .multi_repo import is_git_url, ClonedRepo

        is_git = is_git_url(source)
        if is_git:
            is_container = False
        else:
            is_container = is_container_image(source)

        source_summary = {
            "source_kind": "git-url" if is_git else "container" if is_container else "local-path",
            "status": "in_progress",
            "status_detail": None,
            "assets_discovered": 0,
            "branches_scanned": None,
            "last_generated_at": None,
            "errors": [],
            "source_name": str(source),
            "source_path": str(source),
        }
        source_outcomes[source] = source_summary

        if is_git:
            try:
                clone_ctx = ClonedRepo(source)
                cloned_path = clone_ctx.__enter__()
                clone_managers.append(clone_ctx)
                path_to_analyze = cloned_path
            except RuntimeError as exc:
                message = f"Clone failed: {exc}"
                console.print(f"[red]{message}[/]")
                _record_analysis_error(
                    run_errors,
                    source_summary,
                    source,
                    message,
                    severity="fatal",
                )
                continue
        else:
            path_to_analyze = Path(source)

        if is_container:
            logging.info(f"Source '{source}' detected as a container image.")
            extraction = extract_source_from_image(source, llm_config=llm_config, tier=container_tier)
            if extraction.error or extraction.extracted_dir is None:
                message = f"Error extracting from container image: {extraction.error or 'unknown'}"
                logging.error(message)
                _record_analysis_error(
                    run_errors,
                    source_summary,
                    source,
                    message,
                    severity="fatal",
                )
                continue
            temp_dir = str(extraction.extracted_dir)
            path_to_analyze = extraction.extracted_dir

        if is_container:
            source_summary["source_path"] = "/app" if path_to_analyze.name == "app" else str(path_to_analyze)
            source_summary["source_name"] = str(source)
        else:
            source_summary["source_path"] = str(path_to_analyze.resolve())
            from .reporters.json_reporter import _friendly_source_name
            source_summary["source_name"] = _friendly_source_name(str(path_to_analyze.resolve()))

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

        from .scan_pipeline import ScanPipeline

        scan_path = str(path_to_analyze)

        if skip_unchanged and not is_container and (path_to_analyze / ".git").exists():
            from .incremental import OrgCache

            org_cache = OrgCache()
            cached_sr = org_cache.get_cached(str(path_to_analyze.resolve()))
            if cached_sr is not None:
                console.print(
                    f"[green]Org cache hit[/] for {source} "
                    f"(~/.aibom/cache/org)"
                )
                merged_components: list = []
                merged_rels: list = []
                for s in cached_sr.sources:
                    merged_components.extend(s.components)
                    merged_rels.extend(s.relationships)
                all_analysis_outputs[source] = {
                    "_v2": True,
                    "components": merged_components,
                    "relationships": merged_rels,
                    "_agentic_risk_flags": [],
                    "_agentic_candidate_count": 0,
                }
                source_summary["assets_discovered"] = len(merged_components)
                source_summary["last_generated_at"] = _utcnow_iso()
                if source_summary["status"] == "in_progress":
                    source_summary["status"] = "completed"
                if temp_dir:
                    shutil.rmtree(temp_dir)
                continue

        _scan_cache_hit = False
        if scan_cache_dir:
            from .scan_cache import cache_key, load_cached, save_cached

            _ck = cache_key([scan_path], scan_cache_settings)
            cached = load_cached(
                scan_cache_dir,
                _ck,
                search_dirs=scan_cache_read_dirs,
            )
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
            fail_on=fail_on_severity,
            min_severity=severity_filter,
            strict=strict,
            agentic_scope=agentic_scope,
            agentic_batch_size=agentic_batch_size,
            agentic_concurrency=agentic_concurrency,
            agentic_fast_model=agentic_fast_model,
            agentic_timeout=agentic_timeout,
            agentic_cache_dir=agentic_cache_dir,
            include_code_snippets=include_code_snippets,
        )
        result = _run_pipeline_with_progress(source, pipeline, progress)

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

        if scan_cache_dir and not _scan_cache_hit:
            from .scan_cache import cache_key, save_cached

            _ck = cache_key([scan_path], scan_cache_settings)
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
            save_cached(scan_cache_dir, _ck, _serializable)

        if skip_unchanged and not is_container and (path_to_analyze / ".git").exists():
            from .incremental import OrgCache
            from .models import ScanResult, SourceResult

            org_cache = OrgCache()
            org_cache.store(
                str(path_to_analyze.resolve()),
                ScanResult(
                    metadata=run_metadata,
                    sources=[
                        SourceResult(
                            path=scan_path,
                            components=result.components,
                            relationships=result.relationships,
                        )
                    ],
                    errors=[],
                ),
            )

        source_summary["assets_discovered"] = len(result.components)
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

    from .models import (
        AIComponent as V2Component,
        AIComponentType,
        ComponentRelationship as V2Relationship,
        ScanResult,
        SourceResult,
    )
    from .finding_annotations import annotate_findings
    from .models.scan import RiskFlag
    from .risk import RiskScorer

    v2_sources = []
    for source_path, output in all_analysis_outputs.items():
        if isinstance(output, dict) and output.get("_v2"):
            comps = [
                V2Component.model_validate(c) if isinstance(c, dict) else c
                for c in output["components"]
            ]
            rels = [
                V2Relationship.model_validate(r) if isinstance(r, dict) else r
                for r in output["relationships"]
            ]
            comps, rels, _ = annotate_findings(
                comps,
                rels,
                [],
                include_code_snippets=include_code_snippets,
            )
            v2_sources.append(SourceResult(
                path=source_path,
                components=comps,
                relationships=rels,
            ))
    run_metadata["source_outcomes"] = source_outcomes
    scan_result = ScanResult(
        metadata=run_metadata,
        sources=v2_sources,
        errors=[e.get("message", str(e)) for e in run_errors],
    )

    scorer = RiskScorer()
    scan_result.risk = scorer.score(scan_result)

    for output in all_analysis_outputs.values():
        if isinstance(output, dict):
            for rf in output.get("_agentic_risk_flags", []):
                if isinstance(rf, RiskFlag):
                    scan_result.risk.add_flag(rf)
                elif isinstance(rf, dict):
                    scan_result.risk.add_flag(RiskFlag.model_validate(rf))

    _, _, annotated_risk_flags = annotate_findings(
        [],
        [],
        scan_result.risk.flags,
        include_code_snippets=include_code_snippets,
    )
    scan_result.risk.flags = annotated_risk_flags

    if compliance:
        from .compliance import ComplianceFramework, evaluate_compliance, parse_compliance_cli_value

        parsed = parse_compliance_cli_value(compliance)
        frameworks = list(ComplianceFramework) if parsed == "all" else [parsed]
        for fw in frameworks:
            report = evaluate_compliance(scan_result, fw)
            ctable = Table(
                title=f"Compliance — {fw.value}",
                box=box.MINIMAL_DOUBLE_HEAD,
                header_style="bold cyan",
            )
            ctable.add_column("ID", style="dim")
            ctable.add_column("Requirement")
            ctable.add_column("Status")
            ctable.add_column("Detail")
            for row in report.results:
                st = row.status
                st_style = "green" if st == "pass" else "yellow" if st == "not_applicable" else "red"
                ctable.add_row(
                    row.requirement_id,
                    row.title,
                    f"[{st_style}]{st}[/{st_style}]",
                    row.message,
                )
            console.print(ctable)
            summ = report.summary
            sum_table = Table(box=box.SIMPLE, title=f"Summary — {fw.value}")
            sum_table.add_column("Metric")
            sum_table.add_column("Value", justify="right")
            sum_table.add_row("Total requirements", str(summ["total_requirements"]))
            sum_table.add_row("Passed", str(summ["passed"]))
            sum_table.add_row("Failed", str(summ["failed"]))
            sum_table.add_row("Not applicable", str(summ["not_applicable"]))
            sum_table.add_row("Coverage %", f"{summ['coverage_pct']:.1f}")
            console.print(sum_table)

    if policy:
        from .policy import evaluate_policy, load_policy

        pol = load_policy(Path(policy))
        pr = evaluate_policy(pol, scan_result)
        if pr.violations:
            vtable = Table(
                "Rule",
                "Severity",
                "Message",
                title="Policy violations",
                header_style="bold red",
                box=box.MINIMAL_DOUBLE_HEAD,
            )
            for v in pr.violations:
                extra = ""
                if v.component_name:
                    extra = f" ({v.component_name})"
                if v.file_path:
                    extra += f" @ {v.file_path}"
                vtable.add_row(
                    v.rule,
                    v.severity.value,
                    v.message + extra,
                )
            console.print(vtable)
        if not pr.passed:
            console.print("[bold red]Policy check failed.[/]")
            raise typer.Exit(code=1)

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

    if fail_on_severity and scorer.should_fail(scan_result.risk, fail_on_severity):
        console.print(
            f"[bold red]Risk threshold exceeded: {scan_result.risk.severity.value} "
            f">= {fail_on_severity.value}[/]"
        )
        raise typer.Exit(code=2)

    if post_url and output_format == "json":
        import io as _io

        json_rep = get_reporter("json")
        if json_rep:
            buf = _io.StringIO()
            json_rep.render(scan_result, buf)
            report_data = json.loads(buf.getvalue())
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

    if output_format == "api":
        logging.info("--- Starting API Server ---")
        component_map = {
            source: getattr(output, "components", output) for source, output in all_analysis_outputs.items()
        }
        start_api_server(component_map)

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


@app.command("watch")
def watch_command(
    sources: List[str] = typer.Argument(
        ...,
        help="Source directories or files to poll and re-scan (v2 pipeline).",
    ),
    interval: float = typer.Option(
        2.0,
        "--interval",
        help="Seconds between filesystem polls.",
    ),
    debounce: float = typer.Option(
        0.5,
        "--debounce",
        help="Seconds to wait after a change before re-scanning (coalesces rapid edits).",
    ),
) -> None:
    """Poll paths for changes and re-run the v2 scan, printing component deltas."""
    resolved = [str(Path(s).resolve()) for s in sources]
    for p in resolved:
        if not Path(p).exists():
            console.print(f"[red]Path not found:[/] {p}")
            raise typer.Exit(code=1)

    from .scan_pipeline import ScanPipeline
    from .watch import watch_loop

    def scan_fn() -> object:
        pipeline = ScanPipeline(scan_paths=resolved, output_format="json")
        return pipeline.run()

    try:
        watch_loop(
            sources,
            scan_fn,
            console=console,
            interval=interval,
            debounce=debounce,
        )
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/]")


benchmark_app = typer.Typer(
    help="Compare scan output against ground-truth YAML.",
    no_args_is_help=True,
)
app.add_typer(benchmark_app, name="benchmark")


@benchmark_app.command("run")
def benchmark_run(
    gt: Path = typer.Option(
        ...,
        "--gt",
        exists=True,
        readable=True,
        dir_okay=False,
        resolve_path=True,
        help="Ground-truth YAML file.",
    ),
    scan: Path = typer.Option(
        ...,
        "--scan",
        exists=True,
        readable=True,
        dir_okay=False,
        resolve_path=True,
        help="Scan report JSON (ScanResult or legacy aibom_analysis wrapper).",
    ),
    strict_names: bool = typer.Option(
        False,
        "--strict-names",
        help="Match listed names (case-insensitive) when GT provides names.",
    ),
    fmt: str = typer.Option(
        "table",
        "--format",
        help="Output: table, json, or csv.",
    ),
) -> None:
    from pydantic import ValidationError

    from .benchmark import benchmark_scan, load_ground_truth, render_benchmark_result
    from .diff import load_scan_result_json

    try:
        ground = load_ground_truth(gt)
    except (OSError, ValueError) as exc:
        console.print(f"[red]Failed to load ground truth:[/] {exc}")
        raise typer.Exit(code=1)

    try:
        scan_result = load_scan_result_json(scan)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Invalid JSON:[/] {exc}")
        raise typer.Exit(code=1)
    except ValidationError as exc:
        console.print(f"[red]Invalid scan report:[/] {exc}")
        raise typer.Exit(code=1)
    except OSError as exc:
        console.print(f"[red]Failed to read scan:[/] {exc}")
        raise typer.Exit(code=1)

    allowed = {"table", "json", "csv"}
    if fmt.lower() not in allowed:
        console.print(f"[red]Invalid --format {fmt!r}.[/] Use: {', '.join(sorted(allowed))}")
        raise typer.Exit(code=1)

    result = benchmark_scan(ground, scan_result, strict_names=strict_names)
    render_benchmark_result(result, fmt.lower(), console=console)


kb_app = typer.Typer(help="Manage the AIBOM knowledge base.", no_args_is_help=True)
app.add_typer(kb_app, name="kb")


# ---------------------------------------------------------------------------
# cisco-aibom cache  — manage scan result cache
# ---------------------------------------------------------------------------

cache_app = typer.Typer(help="Manage AIBOM cache entries.", no_args_is_help=True)
app.add_typer(cache_app, name="cache")


@cache_app.command("clear")
def cache_clear(
    cache_dir: Path = typer.Option(
        resolve_cache_root(),
        "--cache-dir",
        help="Cache root directory.",
    ),
    include_agentic: bool = typer.Option(
        True,
        "--include-agentic/--no-agentic",
        help="Also clear the agentic enrichment result cache.",
    ),
) -> None:
    """Remove all cached scan results (and optionally agentic results)."""
    import shutil

    from .scan_cache import clear_cache as _clear

    cache_root = resolve_cache_root(cache_dir)
    removed = 0
    seen_dirs: set[str] = set()
    for directory in cache_read_dirs("scan", cache_root):
        key = str(directory)
        if key in seen_dirs:
            continue
        seen_dirs.add(key)
        removed += _clear(directory)
    console.print(
        f"[green]Removed {removed} cached scan result(s) from {cache_root}[/]"
    )

    if include_agentic:
        count = 0
        for agentic_dir in cache_read_dirs("agentic", cache_root):
            if agentic_dir.exists():
                count += sum(1 for _ in agentic_dir.glob("*.json"))
                shutil.rmtree(agentic_dir)
        if count:
            console.print(
                f"[green]Removed {count} agentic cache file(s) from {cache_root}[/]"
            )
        else:
            console.print("[dim]No agentic cache to clear.[/]")


@cache_app.command("list")
def cache_list(
    cache_type: str = typer.Option(
        "scan",
        "--type",
        help="Cache family: scan, agentic, org, model, packages.",
    ),
    cache_dir: Path = typer.Option(
        resolve_cache_root(),
        "--cache-dir",
        help="Cache root directory.",
    ),
) -> None:
    """List cached entries for one cache family."""
    from rich.table import Table

    from .cache_inspector import list_cache_entries

    if cache_type not in cache_types():
        console.print(
            f"[red]Unsupported cache type:[/] {cache_type}. "
            f"Use one of: {', '.join(cache_types())}."
        )
        raise typer.Exit(code=1)

    entries = list_cache_entries(cache_type, cache_dir)
    if not entries:
        console.print(f"[dim]No cached {cache_type} entries found.[/]")
        return

    table = Table(title=f"{cache_type.title()} Cache ({resolve_cache_root(cache_dir)})")
    table.add_column("Entry", no_wrap=True)
    table.add_column("Subtype")
    table.add_column("Cached At")
    table.add_column("Size", justify="right")
    table.add_column("Detail")
    for e in entries:
        table.add_row(
            str(e["id"]),
            str(e.get("subtype", "")),
            str(e.get("cached_at", "unknown")),
            str(e.get("size", "")),
            str(e.get("detail", "")),
        )
    console.print(table)


@cache_app.command("get")
def cache_get(
    cache_type: str = typer.Argument(..., help="Cache family to inspect."),
    entry_ref: str = typer.Argument(..., help="Entry id, prefix, or logical reference."),
    cache_dir: Path = typer.Option(
        resolve_cache_root(),
        "--cache-dir",
        help="Cache root directory.",
    ),
    sha: Optional[str] = typer.Option(
        None,
        "--sha",
        help="Commit SHA for org cache lookups.",
    ),
    model_id: Optional[str] = typer.Option(
        None,
        "--model-id",
        help="Optional model id filter for model cache lookups.",
    ),
    raw_json: bool = typer.Option(
        False,
        "--raw-json",
        help="Print the raw cache payload instead of a summary.",
    ),
) -> None:
    """Inspect a specific cache entry by type."""
    from .cache_inspector import get_cache_entry

    if cache_type not in cache_types():
        console.print(
            f"[red]Unsupported cache type:[/] {cache_type}. "
            f"Use one of: {', '.join(cache_types())}."
        )
        raise typer.Exit(code=1)

    try:
        entry = get_cache_entry(
            cache_type,
            entry_ref,
            cache_root=cache_dir,
            sha=sha,
            model_id=model_id,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1)

    if raw_json:
        console.print(Syntax(json.dumps(entry["payload"], indent=2), "json", theme="monokai"))
        return

    summary = Table(title=f"{cache_type.title()} Cache Entry", box=box.SIMPLE_HEAVY)
    summary.add_column("Field")
    summary.add_column("Value")
    summary.add_row("Entry", str(entry["id"]))
    summary.add_row("Subtype", str(entry.get("subtype", "")))
    summary.add_row("Path", str(entry.get("path", "")))
    summary.add_row("Cached At", str(entry.get("cached_at", "unknown")))
    summary.add_row("Size", str(entry.get("size", "")))
    summary.add_row("Detail", str(entry.get("detail", "")))
    if entry.get("repo_path"):
        summary.add_row("Repo Path", str(entry["repo_path"]))
    if entry.get("sha"):
        summary.add_row("SHA", str(entry["sha"]))
    console.print(summary)
    if entry.get("repo_path"):
        console.print(f"[cyan]Repo Path:[/] {entry['repo_path']}", soft_wrap=True)
    if entry.get("sha"):
        console.print(f"[cyan]SHA:[/] {entry['sha']}", soft_wrap=True)

    payload = entry["payload"]
    if cache_type == "scan":
        component_names = [
            str(item.get("name"))
            for item in payload.get("components", [])
            if isinstance(item, dict) and item.get("name")
        ]
        if component_names:
            console.print(
                f"[cyan]Components:[/] {', '.join(component_names[:10])}"
            )
    elif cache_type == "model":
        models = payload.get("models", payload)
        if isinstance(models, dict) and models:
            console.print(
                f"[cyan]Models:[/] {', '.join(list(models.keys())[:10])}"
            )
    elif cache_type == "packages":
        if payload.get("summary"):
            console.print(f"[cyan]Summary:[/] {payload['summary']}")


# ---------------------------------------------------------------------------
# cisco-aibom plugin  — discover and manage plugins
# ---------------------------------------------------------------------------

plugin_app = typer.Typer(help="Discover and manage AIBOM plugins.", no_args_is_help=True)
app.add_typer(plugin_app, name="plugin")

def _diff_impl(old_report: Path, new_report: Path, fmt: str) -> None:
    allowed = {"table", "json", "markdown"}
    if fmt.lower() not in allowed:
        console.print(
            f"[red]Invalid --format {fmt!r}.[/] "
            "Use: table, json, markdown.",
        )
        raise typer.Exit(code=1)
    from pydantic import ValidationError

    from .diff import diff_scan_results, load_scan_result_json, render_diff

    try:
        old_sr = load_scan_result_json(old_report)
        new_sr = load_scan_result_json(new_report)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Invalid JSON:[/] {exc}")
        raise typer.Exit(code=1)
    except ValidationError as exc:
        console.print(f"[red]Invalid report structure:[/] {exc}")
        raise typer.Exit(code=1)
    except OSError as exc:
        console.print(f"[red]Failed to read report:[/] {exc}")
        raise typer.Exit(code=1)

    result = diff_scan_results(old_sr, new_sr)
    try:
        render_diff(result, fmt, console)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1)


diff_app = typer.Typer(
    help="Compare two AIBOM JSON scan reports.",
    no_args_is_help=True,
)
app.add_typer(diff_app, name="diff")


@diff_app.command("run")
def diff_run(
    old_report: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Path to the older JSON report.",
    ),
    new_report: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Path to the newer JSON report.",
    ),
    fmt: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, json, or markdown.",
    ),
) -> None:
    """Compare reports; use when you need ``--format`` after the report paths (Typer quirk)."""
    _diff_impl(old_report, new_report, fmt)


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
