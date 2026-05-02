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

"""AIBOM middleware for the Deep Agents harness.

``AIBOMScannerMiddleware`` post-processes the agent's final message,
extracts structured AIBOM findings from the JSON output, and converts
them into ``AIComponent`` / ``ComponentRelationship`` / ``RiskFlag``
objects that merge into the deterministic ``ScanResult``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from ..models import (
    AIComponent,
    AIComponentType,
    CodeSnippet,
    ComponentRelationship,
    DetectionSource,
    DecisionAnnotation,
    EvidenceLocation,
    RelationshipType,
    RiskFlag,
    Severity,
)

_LOGGER = logging.getLogger(__name__)


def _ckey(c: AIComponent) -> tuple[str, str]:
    """Consolidation key matching ``scan_pipeline._consolidation_key``."""
    canonical = (c.model_name or c.name).lower().strip()
    return (canonical, c.component_type.value)


_IID_TRAILING_LINE_RE = re.compile(r"^(?P<head>.+)_(?P<line>\d+)$")


def _parse_iid_name_prefix(iid: str) -> str | None:
    """Extract the ``name`` prefix from an instance_id of the canonical form
    ``"<name>_<absolute_path>_<line>"``.

    ``AIComponent.instance_id`` is built as
    ``f"{name}_{file_path}_{line_number}"`` (see
    ``aibom.models.scan.AIComponent.model_post_init``). ``file_path`` is
    always an absolute POSIX path that begins with ``/``, so the last
    occurrence of ``"_/"`` in ``iid`` (after stripping the trailing
    ``_<digits>``) is a reliable boundary between name and path even
    when the name itself contains underscores or the path contains
    underscores. Naive ``rsplit("_", 1)`` would mis-split paths that
    contain ``_`` (e.g. ``foo_bar.py``).

    Returns ``None`` when ``iid`` does not match the canonical format
    (no trailing line number or no embedded absolute path); callers must
    treat that as "unparseable, drop".
    """
    if not iid:
        return None
    m = _IID_TRAILING_LINE_RE.match(iid)
    if not m:
        return None
    head = m.group("head")
    sep_idx = head.rfind("_/")
    if sep_idx < 0:
        return None
    return head[:sep_idx]


_CLASS_NAME_RE = re.compile(
    r"^[A-Z][a-zA-Z0-9]*$"
)


def _is_class_name_not_model_id(name: str) -> bool:
    """Return True when *name* looks like a Python/Go class name, not a model ID.

    Real model identifiers contain slashes (``meta-llama/Llama-3``),
    hyphens (``gpt-4o``), version-like dot segments (``3.5-turbo``),
    or are entirely lowercase.  CamelCase identifiers such as
    ``OpenAILLM`` or ``ChatOpenAI`` are wrapper classes.
    """
    if not name or len(name) < 2:
        return False
    if "/" in name or ":" in name or "-" in name:
        return False
    if re.search(r"\d+\.\d+", name):
        return False
    if not _CLASS_NAME_RE.match(name):
        return False
    has_upper_after_first = any(c.isupper() for c in name[1:])
    return has_upper_after_first


_ENDPOINT_TYPES = frozenset({
    AIComponentType.LLM_ENDPOINT,
    AIComponentType.MODEL_ENDPOINT,
    AIComponentType.VECTOR_STORE,
})


# --- Hallucination guardrails ------------------------------------------------
#
# The agentic stage occasionally fabricates values that do not appear in the
# scanned sources: most commonly a synthesized endpoint URL (derived from SDK
# docstrings, validation errors, or provider hints) and invented metadata
# keys used to justify the fabrication. Three middleware gates defend the
# final BOM against this:
#
# 1. ``_sanitize_metadata``            — strips metadata keys not present in
#                                        the documented schema surfaced in
#                                        ``prompts.py``. Catches keys such as
#                                        ``resolution``, ``llm_notes``, or
#                                        ``inferred_from`` which only a
#                                        hallucinating LLM would emit.
# 2. ``_rewrite_if_ungrounded_endpoint`` — for endpoint-typed components that
#                                        carry an ``env_var`` marker, verify
#                                        the URL literal appears in live code
#                                        (docstrings and comments stripped).
#                                        Rewrites fabrications to the
#                                        ``env:<VAR>`` placeholder form.
# 3. ``_cap_confidence_if_unresolved`` — caps ``heuristic_confidence`` at 0.5
#                                        for any env-backed agentic component
#                                        that still lacks a concrete value.

_ALLOWED_METADATA_KEYS: frozenset[str] = frozenset({
    # Provenance / config source
    "env_var",
    "env",
    "env_context",
    "env_value",
    "config_kind",
    "config_key",
    "source",
    "source_file",
    "resolved_value",
    "resolved_from",
    "file_loaded_limitation",
    "redacted",
    "section",
    "scanner",
    # Model registry enrichment
    "model_family",
    "model_provider",
    "model_name",
    "resolved_model",
    "provider",
    "provider_name",
    "family",
    "mode",
    "license",
    "deprecated",
    "model_card_url",
    "context_length",
    "registry_source",
    "detection_method",
    # Endpoint-specific
    "endpoint_url",
    "endpoint_status",
    "provider_domain",
    "region",
    "deployment_id",
    # Relationship / scoping
    "framework",
    "service",
    "service_name",
    "helm_key",
    "helm_chart",
    "chart_path",
    "kubernetes_kind",
    "graph_id",
    "graph_spec",
    "node_type",
    "agent_name",
    "task_name",
    "tool_name",
    "agent",
    "job",
    "action",
    # Docker / K8s
    "image",
    "container_image",
    "kind",
    "annotation",
    # Cloud / IaC
    "cloud_provider",
    "instance_type",
    "gpu_count",
    "logical_id",
    "resource_name",
    "location",
    "creation_time",
    "size_bytes",
    "format",
    "field",
    "arm_deployment",
    "arm_parameter",
    "arm_sku",
    "arm_type",
    "bicep_param",
    "bicep_type",
    "cfn_parameter",
    "cfn_property",
    "cfn_type",
    "terraform_local",
    "terraform_variable",
    "cicd_type",
    # Agent / MCP
    "agent_card",
    "agent_evidence",
    "skills",
    "remote_verification",
    "mcp_tool_name",
    "mcp_decorators",
    "qualified_name",
    "protocol_match_count",
    "class_name",
    "constructor",
    "patterns",
    # Dependency / package
    "ecosystem",
    "manifest",
    "package",
    "package_type",
    "package_summary",
    "package_keywords",
    "package_classifiers",
    "version_spec",
    "known_ai_package",
    "installed_path",
    "local",
    "local_path",
    "declared_in_manifests",
    "import_found",
    # Vulnerability scanner enrichment
    "vulnerabilities",
    "risk_flag",
    # Call-pattern / KB provenance (LLM may echo these from deterministic rows)
    "call_pattern",
    "tool_kwarg",
    "tool_decorator",
    "tool_conversion",
    "assigned_to",
    "prompt_kwarg",
    "enclosing_call",
    "variable_ref",
    "kb_id",
    "partial_kb_id",
    "observation_type",
    "obs_module",
    "import_statement",
    "extracted_from_kwarg",
    "suggestive_signal",
    "parent_dir",
    "cache_ai_co_occurrence",
    "cache_library",
    "shadow_ai",
    # Secret detector
    "secret_source",
    "secret_path",
    "secret_name",
    "secret_ref",
    "secret_type",
    "has_vault_import",
    # Deployment detector
    "store_technology",
    "backend_selector_key",
    "backend_selector_value",
    # Structural agent
    "discovery",
    "structural_signature_id",
    "react_loop_start_line",
    "react_loop_end_line",
    "react_loop_rationale",
    "class_start_line",
    "class_end_line",
})


def _sanitize_metadata(
    meta: Any,
    *,
    component_name: str,
    component_type: str,
) -> dict[str, Any]:
    """Strip metadata keys absent from the documented LLM schema.

    The prompt section "Metadata schema — allow-list only" tells the LLM
    exactly which keys are valid. Any other key is a signal the LLM has
    invented extra structure to justify a hallucinated component. We drop
    unknown keys and emit a ``WARNING`` so the hallucination is visible
    in logs without dropping the entire component.
    """
    if not isinstance(meta, dict):
        return {}
    cleaned: dict[str, Any] = {}
    for k, v in meta.items():
        if not isinstance(k, str):
            continue
        if k in _ALLOWED_METADATA_KEYS:
            cleaned[k] = v
            continue
        _LOGGER.warning(
            "Stripped unknown metadata key %r from agentic %s component %r "
            "(not in documented schema — likely hallucinated)",
            k, component_type, component_name,
        )
    return cleaned


_DOCSTRING_RE = re.compile(r'("""|\'\'\')(.*?)\1', re.DOTALL)
_BLOCK_COMMENT_RE = re.compile(r'/\*.*?\*/', re.DOTALL)
# Line comments must start at the line (possibly after whitespace). This
# avoids wrongly treating ``://`` inside URLs as the start of a ``//``
# comment (e.g. ``https://api.example.com``).
_LINE_COMMENT_PREFIXES: tuple[str, ...] = ("#", "//")

_CONFIG_FILE_SUFFIXES: frozenset[str] = frozenset({
    ".yaml", ".yml", ".json", ".toml", ".ini", ".env", ".cfg",
    ".properties", ".conf",
})

_CODE_FILE_SUFFIXES: frozenset[str] = frozenset({
    ".py", ".pyi",
    ".js", ".jsx", ".mjs", ".ts", ".tsx",
    ".go", ".java", ".kt", ".kts",
    ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp",
    ".rs", ".rb", ".php", ".sh", ".bash", ".swift",
})

_CHECKABLE_BARE_NAMES: frozenset[str] = frozenset({
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
})


def _strip_docstrings_and_comments(text: str, suffix: str) -> str:
    """Strip docstrings and comments from *text* for grounding checks.

    Leaves YAML/JSON/TOML/INI untouched: those are config, not docs.
    For Python, remove triple-quoted docstring/string literals. For
    C-family and Go-family, strip block and line comments. Line
    comments are only stripped when the line begins with ``#`` or
    ``//`` (after optional whitespace) — this preserves URLs such as
    ``https://...`` that legitimately contain ``//`` mid-line. Inline
    trailing comments may be preserved; that is an accepted trade-off
    in favour of not falsely de-grounding real code.
    """
    if suffix in _CONFIG_FILE_SUFFIXES:
        return text
    t = _DOCSTRING_RE.sub("", text)
    if suffix in {
        ".js", ".jsx", ".mjs", ".ts", ".tsx",
        ".go", ".java", ".kt", ".kts",
        ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp",
        ".rs", ".swift",
    }:
        t = _BLOCK_COMMENT_RE.sub("", t)
    kept_lines: list[str] = []
    for line in t.split("\n"):
        stripped = line.lstrip()
        is_comment_line = any(
            stripped.startswith(prefix) for prefix in _LINE_COMMENT_PREFIXES
        )
        if is_comment_line:
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def _url_grounded_in_live_code(
    url: str,
    *,
    allowed_roots: list[str],
    max_files: int = 500,
) -> bool:
    """Return True iff *url* appears literally in non-comment source
    under any of *allowed_roots*.

    Bounded to ``max_files`` inspected files so the check is cheap even
    on large monorepos. The gate only fires for a small number of
    endpoint components each scan, so this budget is not a hot path.
    """
    if not url or not allowed_roots:
        return False
    checked = 0
    for root in allowed_roots:
        try:
            root_path = Path(root)
        except OSError:
            continue
        if not root_path.exists():
            continue
        if root_path.is_file():
            candidates = [root_path]
        else:
            candidates = list(root_path.rglob("*"))
        for p in candidates:
            try:
                if not p.is_file():
                    continue
            except OSError:
                continue
            suffix = p.suffix.lower()
            if (
                suffix not in _CODE_FILE_SUFFIXES
                and suffix not in _CONFIG_FILE_SUFFIXES
                and p.name not in _CHECKABLE_BARE_NAMES
            ):
                continue
            checked += 1
            if checked > max_files:
                return False
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            stripped = _strip_docstrings_and_comments(text, suffix)
            if url in stripped:
                return True
    return False


def _rewrite_if_ungrounded_endpoint(
    comp: AIComponent,
    *,
    allowed_roots: list[str],
) -> AIComponent:
    """Rewrite endpoint components whose URL was fabricated by the LLM.

    Preconditions: agentic-source component, endpoint type, and
    ``metadata.env_var`` or ``metadata.env`` present (indicating the
    URL ought to derive from an environment variable resolution).
    If the ``name`` is an ``http://``/``https://`` URL but that URL
    does not appear verbatim in live code, rewrite the component to the
    ``env:<VAR>`` placeholder shape and drop any endpoint URL claim.
    """
    if comp.component_type not in _ENDPOINT_TYPES:
        return comp
    meta = comp.metadata or {}
    env_var = meta.get("env_var") or meta.get("env")
    if not isinstance(env_var, str) or not env_var:
        return comp
    name = comp.name or ""
    name_l = name.lower()
    if not (name_l.startswith("http://") or name_l.startswith("https://")):
        return comp
    if _url_grounded_in_live_code(name, allowed_roots=allowed_roots):
        return comp
    _LOGGER.warning(
        "Rewrote ungrounded endpoint URL %r → env:%s for %s component "
        "(URL not found in live code under scan roots; likely hallucination)",
        name, env_var, comp.component_type.value,
    )
    new_meta = dict(meta)
    new_meta.pop("endpoint_url", None)
    new_meta["env_var"] = env_var
    capped = min(comp.heuristic_confidence or 1.0, 0.5)
    return comp.model_copy(update={
        "name": f"env:{env_var}",
        "model_name": None,
        "heuristic_confidence": capped,
        "metadata": new_meta,
    })


_CONFIDENCE_PLACEHOLDER_RE = re.compile(r"\$\{[^}]+\}")


def _cap_confidence_if_unresolved(comp: AIComponent) -> AIComponent:
    """Cap ``heuristic_confidence`` at 0.5 for unresolved env-backed rows.

    Any component that carries an ``env_var`` (or Dockerfile ``env``)
    marker but still lacks a concrete value is an unresolved placeholder.
    The prompt's "Confidence calibration" rule requires ≤0.5 for these;
    the middleware enforces it regardless of what the LLM emitted.
    """
    meta = comp.metadata or {}
    env_var = meta.get("env_var") or meta.get("env")
    if not isinstance(env_var, str) or not env_var:
        return comp
    has_concrete_value = False
    if (
        isinstance(comp.model_name, str)
        and comp.model_name
        and not _CONFIDENCE_PLACEHOLDER_RE.search(comp.model_name)
        and not comp.model_name.startswith("env:")
    ):
        has_concrete_value = True
    name_l = (comp.name or "").lower()
    if name_l.startswith("http://") or name_l.startswith("https://"):
        has_concrete_value = True
    if has_concrete_value:
        return comp
    current_conf = (
        comp.heuristic_confidence
        if comp.heuristic_confidence is not None
        else 1.0
    )
    if current_conf <= 0.5:
        return comp
    _LOGGER.warning(
        "Capped heuristic_confidence %.2f → 0.5 for unresolved env-backed "
        "%s component %r (env_var=%s, no concrete value)",
        current_conf, comp.component_type.value, comp.name, env_var,
    )
    return comp.model_copy(update={"heuristic_confidence": 0.5})


_HELM_VALUES_FILE_SUFFIXES: tuple[str, ...] = (
    "values.yaml", "values.yml",
    "chart.yaml", "chart.yml",
)


def _should_reject_tool_from_helm(comp: AIComponent) -> tuple[bool, str]:
    """Return ``(reject, reason)`` when a ``tool`` component is actually a
    Helm / Kubernetes service, not an agent-callable tool.

    Matching heuristics (any one is sufficient):

    * ``framework`` is ``helm``.
    * ``metadata.service_name``, ``metadata.helm_key``,
      ``metadata.helm_chart``, ``metadata.chart_path`` or
      ``metadata.kubernetes_kind`` is set.
    * ``file_path`` ends in ``values.yaml`` / ``chart.yaml``.
    * ``file_path`` is under a ``charts/`` directory and ends in
      ``.yaml`` / ``.yml``.
    """
    if comp.component_type != AIComponentType.TOOL:
        return False, ""
    if (comp.framework or "").lower() == "helm":
        return True, "framework=helm"
    meta = comp.metadata or {}
    for key in ("service_name", "helm_key", "helm_chart",
                "chart_path", "kubernetes_kind"):
        v = meta.get(key)
        if isinstance(v, str) and v:
            return True, f"metadata.{key} set ({v!r})"
    fp = (comp.file_path or "").lower()
    if not fp:
        return False, ""
    for suffix in _HELM_VALUES_FILE_SUFFIXES:
        if fp.endswith(suffix):
            return True, f"file_path ends with {suffix!r}"
    if "/charts/" in fp and fp.endswith((".yaml", ".yml")):
        return True, "file_path under charts/ directory"
    return False, ""


# --- Deterministic-model removal guard --------------------------------------
#
# Private/self-hosted inference deployments (vLLM, TGI, Triton, BentoML,
# SageMaker) and cloud provider deployment aliases (Azure OpenAI deployments,
# Bedrock inference profiles, custom SKUs) surface through ``.yaml`` /
# ``values.yaml`` / ``.env`` / Dockerfile config with names that are *unique
# to each customer environment* — e.g. ``org/custom-model/stable`` or
# ``prod-chat-gpt4o-westus``. They cannot be pre-catalogued in any registry.
#
# Once the deterministic config scan pulls such a string, the agent MUST NOT
# remove it merely because ``lookup_model`` returns ``found: false``. The
# prompt already documents this (see ``prompts.py`` — "Deployment IDs may need
# normalization, not invention"), but the LLM has been observed overriding it
# with registry-verification logic. This guard is the deterministic safety
# rail that prevents ground-truth loss from config-sourced private models.
#
# Scope (all must hold for the guard to fire):
#   * component_type == MODEL
#   * detection_source is any deterministic scanner (not AGENTIC)
#   * name is a concrete string — contains ``/``, ``:``, ``-`` or a version
#     segment (i.e. passes through ``_is_class_name_not_model_id`` as False)
#   * name is not an ``env:<VAR>`` placeholder
#   * removal reason cites a registry / alias / private / unresolved marker

_PROTECTED_REMOVAL_REASON_MARKERS: tuple[str, ...] = (
    "registry",
    "deployment alias",
    "deployment-style",
    "deployment name",
    "private",
    "custom deployment",
    "does not resolve",
    "not resolve",
    "cannot be confirmed",
    "cannot be verified",
    "insufficient evidence",
    "lookup failed",
    "may be private",
)


def _should_protect_deterministic_model_removal(
    comp: AIComponent, reason: str
) -> tuple[bool, str]:
    """Return ``(protect, explanation)`` when the agent tries to remove a
    deterministically-scanned concrete model string due to registry miss.

    See module-level comment above for the full rationale. Returns
    ``(False, "")`` when the removal should be allowed to proceed.
    """
    if comp.component_type != AIComponentType.MODEL:
        return False, ""
    if comp.detection_source == DetectionSource.AGENTIC:
        return False, ""
    name = (comp.name or "").strip()
    if not name:
        return False, ""
    if name.startswith("env:"):
        return False, ""
    if _is_class_name_not_model_id(name):
        return False, ""
    lowered = (reason or "").lower()
    for marker in _PROTECTED_REMOVAL_REASON_MARKERS:
        if marker in lowered:
            return True, (
                f"concrete deterministic model "
                f"(source={comp.detection_source.value}); "
                f"reason marker hit: {marker!r}"
            )
    return False, ""


# --- Sentinel / self-contradiction guards ------------------------------------
#
# The LLM occasionally emits placeholder "names" that explain why nothing was
# added (e.g. ``"USES_MODEL placeholder skipped"``) instead of simply omitting
# the entry. These leak into the BOM as fake components/relationships. The
# prompt explicitly forbids this (see ``prompts.py`` — "Sentinel-free output"),
# and :class:`AIBOMScannerMiddleware` rejects any item whose name or
# justification matches the patterns below as a last-line defense.

_SENTINEL_NAME_SUBSTRINGS: tuple[str, ...] = (
    "placeholder",
    "skipped",
    "omitted",
    "n/a",
    "not applicable",
    "none found",
    "no match",
    "no component",
    "no relationship",
    "no suitable",
    "nothing to add",
    "unknown",
)


_NEGATING_JUSTIFICATION_PREFIXES: tuple[str, ...] = (
    "no ",
    "not ",
    "none ",
    "cannot ",
    "unable ",
    "nothing ",
    "insufficient ",
    "placeholder",
    "skipped",
    "omitted",
    "n/a",
)


def _is_sentinel_name(name: Any) -> bool:
    """Return True when *name* is empty, ``"unknown"``, or contains a
    sentinel substring indicating the LLM meant to omit the entry.

    Matching is case-insensitive and substring-based — the LLM tends to
    produce varied phrasings like ``"USES_MODEL placeholder skipped"``,
    ``"placeholder - skipped"``, or ``"None found"``.
    """
    if not name:
        return True
    if not isinstance(name, str):
        return True
    cleaned = name.strip().lower()
    if not cleaned:
        return True
    if cleaned == "unknown":
        return True
    return any(token in cleaned for token in _SENTINEL_NAME_SUBSTRINGS)


def _is_negating_justification(justification: Any) -> bool:
    """Return True when *justification* begins with a negation such as
    ``"No suitable model found"`` — indicating the LLM itself concluded
    nothing should be added, yet emitted a record anyway.
    """
    if not justification or not isinstance(justification, str):
        return False
    head = justification.strip().lower()
    return any(head.startswith(prefix) for prefix in _NEGATING_JUSTIFICATION_PREFIXES)


_AGENT_CLASSIFICATION_TYPES: frozenset[AIComponentType] = frozenset({
    AIComponentType.AGENT,
    AIComponentType.AGENT_PROXY,
})


_AGENT_CLASSIFICATION_TYPE_STRINGS: frozenset[str] = frozenset(
    {t.value for t in _AGENT_CLASSIFICATION_TYPES}
)


_ACCEPTED_AGENT_EVIDENCE_PATTERNS: frozenset[str] = frozenset({
    "framework_agent",
    "react_loop",
    "framework_inheritance",
    "a2a_server",
    "remote_proxy",
    "other",
})


def _normalize_ws(text: str) -> str:
    """Collapse all runs of whitespace in *text* to a single space.

    Used by :func:`_verify_agent_evidence` so the LLM does not have to
    reproduce the citation snippet byte-for-byte — tab/newline/trailing
    spacing differences are tolerated, but the character content must
    match exactly.
    """
    return " ".join(text.split())


def _verify_agent_evidence(
    raw: Any,
    *,
    allowed_roots: list[str],
) -> tuple[bool, str]:
    """Verify that *raw* is a populated, non-hallucinated ``AgentEvidence``.

    The middleware invokes this for every verdict that classifies a
    component as an agent or agent proxy. Verification is purely offline:
    the ``definition_file`` must exist on disk inside *allowed_roots*, the
    cited line range must be valid, and the (whitespace-normalized)
    ``evidence_snippet`` must appear inside that range.

    Parameters
    ----------
    raw:
        The raw ``agent_evidence`` dict taken verbatim from the agent's
        JSON response. ``None`` or non-dict values fail immediately.
    allowed_roots:
        Paths outside of which file access is disallowed. Mirrors the
        same allow-list used by :meth:`AIBOMScannerMiddleware._read_code_snippet`.

    Returns
    -------
    (ok, reason)
        ``ok`` is ``True`` when every check passes. ``reason`` is an empty
        string on success, otherwise a short, human-readable explanation
        used by the caller for its warning log.
    """
    if not raw:
        return False, "missing agent_evidence"
    if not isinstance(raw, dict):
        return False, "agent_evidence is not an object"

    pattern = raw.get("pattern")
    if pattern not in _ACCEPTED_AGENT_EVIDENCE_PATTERNS:
        return False, f"invalid pattern '{pattern}'"

    definition_file = raw.get("definition_file") or ""
    if not isinstance(definition_file, str) or not definition_file:
        return False, "empty definition_file"

    try:
        resolved = Path(definition_file).resolve()
    except OSError:
        return False, "file path cannot be resolved"
    if allowed_roots:
        inside = False
        for root in allowed_roots:
            try:
                root_path = Path(root).resolve()
            except OSError:
                continue
            if resolved == root_path or root_path in resolved.parents:
                inside = True
                break
        if not inside:
            return False, "file outside allowed roots"

    if not resolved.is_file():
        return False, "file not found"

    try:
        file_text = resolved.read_text(encoding="utf-8")
    except OSError:
        return False, "file not readable"

    lines = file_text.splitlines()
    total = len(lines)
    start = raw.get("definition_start_line") or 0
    end = raw.get("definition_end_line") or 0
    if not isinstance(start, int) or not isinstance(end, int):
        return False, "invalid line range"
    if start < 1 or end < start or end > total:
        return False, "invalid line range"

    snippet = raw.get("evidence_snippet") or ""
    if not isinstance(snippet, str) or not snippet.strip():
        return False, "missing evidence_snippet"

    window = "\n".join(lines[start - 1:end])
    if _normalize_ws(snippet) not in _normalize_ws(window):
        return False, "snippet not found in cited range"

    return True, ""


def _normalize_endpoint_model_name(comp: AIComponent) -> AIComponent:
    """Normalise endpoint components: ``name``=URL, ``model_name``=None.

    An endpoint can host multiple models, so ``model_name`` is left
    ``None`` and model identity is captured via ``HOSTS_MODEL``
    relationships.  The original env-var key is recorded in
    ``metadata.env_var`` for provenance.
    """
    if comp.component_type not in _ENDPOINT_TYPES:
        return comp
    url = (comp.metadata or {}).get("endpoint_url") or ""
    if not url:
        return comp
    meta = dict(comp.metadata)
    if "env_var" not in meta:
        env_key = meta.get("env") or meta.get("config_key") or meta.get("helm_key", "")
        if env_key:
            meta["env_var"] = env_key
    updates: dict[str, Any] = {
        "model_name": None,
        "heuristic_confidence": max(comp.heuristic_confidence, 0.8),
        "metadata": meta,
    }
    if comp.name.startswith("env:") or not comp.name.startswith("http"):
        updates["name"] = url
    return comp.model_copy(update=updates)


def _reject_class_name_models(
    components: list[AIComponent],
) -> list[AIComponent]:
    """Remove ``model`` components whose name is a class name, not a model ID."""
    result: list[AIComponent] = []
    for comp in components:
        if (
            comp.component_type == AIComponentType.MODEL
            and _is_class_name_not_model_id(comp.name)
        ):
            _LOGGER.debug("Class-name model gate: removing %s", comp.name)
            continue
        result.append(comp)
    return result


# Types where the component's ``name`` is supposed to be a concrete
# identifier (a model id or an endpoint URL). For these, an ``env:<VAR>``
# placeholder name carries zero BOM information — a consumer cannot tell
# from ``env:MODEL_ENDPOINT_URL`` which endpoint is actually used.
#
# ``SECRET`` is intentionally NOT here: its ``name`` being the env-var
# name is the whole point — a consumer needs to know the app reads
# ``OPENAI_API_KEY`` to understand which credential is required. The
# secret value itself must never appear anyway.
_ENV_PLACEHOLDER_IDENTIFIER_TYPES: tuple[AIComponentType, ...] = (
    AIComponentType.MODEL,
    AIComponentType.MODEL_ENDPOINT,
    AIComponentType.LLM_ENDPOINT,
)


def _drop_env_placeholder_identifiers(
    components: list[AIComponent],
) -> list[AIComponent]:
    """Remove identifier components whose name is an unresolved ``env:<VAR>``.

    ``env:<VAR>`` is the scanner's placeholder shape for an identifier we
    saw referenced in code (``os.getenv("FOO")``, unresolved ``${FOO}``
    in YAML, Helm templating, Dockerfile ``$FOO``) but whose concrete
    value we could not resolve anywhere in the scanned inputs.

    Covered types (see ``_ENV_PLACEHOLDER_IDENTIFIER_TYPES``):
      * ``MODEL``          — name is supposed to be a model id
        (``gpt-4o``, ``org/custom/stable``).
      * ``MODEL_ENDPOINT`` — name is supposed to be an endpoint URL.
      * ``LLM_ENDPOINT``   — name is supposed to be an endpoint URL.

    For these types an ``env:`` placeholder name tells a BOM consumer
    neither the value nor how to resolve it, so it must not leak into
    the final output. ``SECRET`` is excluded because the env-var name
    IS its primary identifier.

    A component is preserved despite an ``env:`` name if ``model_name``
    holds a concrete (non-placeholder) value, which indicates the
    scanner resolved the referenced variable on a later pass.

    Applied after agentic enrichment so the final BOM cannot leak
    unresolved env placeholders regardless of LLM behaviour.
    """
    result: list[AIComponent] = []
    for comp in components:
        if comp.component_type not in _ENV_PLACEHOLDER_IDENTIFIER_TYPES:
            result.append(comp)
            continue
        name = comp.name or ""
        if not name.startswith("env:"):
            result.append(comp)
            continue
        resolved = comp.model_name
        if isinstance(resolved, str) and resolved and not resolved.startswith("env:"):
            result.append(comp)
            continue
        _LOGGER.info(
            "Env-placeholder gate: dropping unresolved %s component %r "
            "(file=%s:%s)",
            comp.component_type.value, name, comp.file_path, comp.line_number,
        )
    return result


def _remove_unresolved_embedders(
    components: list[AIComponent],
    relationships: list["ComponentRelationship"],
) -> list[AIComponent]:
    """Remove ``embedding`` components without a concrete model identifier."""
    from ..models.enums import RelationshipType

    has_embedding_rel: set[str] = set()
    for rel in relationships:
        if rel.relationship_type == RelationshipType.USES_EMBEDDING:
            has_embedding_rel.add(rel.source_name)

    result: list[AIComponent] = []
    for comp in components:
        if comp.component_type == AIComponentType.EMBEDDING:
            has_model = bool(comp.model_name or comp.embedding_model)
            has_rel = comp.name in has_embedding_rel
            if not has_model and not has_rel:
                _LOGGER.debug("Unresolved embedder gate: removing %s", comp.name)
                continue
        result.append(comp)
    return result


class AIBOMScannerMiddleware:
    """Extracts structured AIBOM data from agent output.

    After the agent finishes, call :meth:`extract_findings` on the final
    message content to obtain components, relationships, and risk flags
    that can be merged into the deterministic scan results.
    """

    def __init__(
        self,
        *,
        include_code_snippets: bool = False,
        allowed_roots: list[str] | None = None,
    ) -> None:
        self.include_code_snippets = include_code_snippets
        self.allowed_roots = allowed_roots or []

    def extract_findings(
        self, agent_output: str
    ) -> tuple[list[AIComponent], list[ComponentRelationship], list[RiskFlag]]:
        """Parse the agent's JSON string output into AIBOM model objects."""
        data = self._parse_json(agent_output)
        if data is None:
            return [], [], []
        return self.extract_findings_from_dict(data)

    def extract_findings_from_dict(
        self, data: dict[str, Any]
    ) -> tuple[list[AIComponent], list[ComponentRelationship], list[RiskFlag]]:
        """Extract findings from an already-parsed dict."""
        components = self._extract_new_components(data)
        enrichments = self._extract_enrichments(data)
        relationships = self._extract_relationships(data)
        relationships = self._drop_dependency_mcp_client_rels(relationships)
        relationships = self._drop_dep_to_dep_uses_model(relationships)
        risk_flags = self._extract_risk_flags(data)
        all_components = self._enforce_embedding_resolution(
            components + enrichments, relationships,
        )
        return all_components, relationships, risk_flags

    def apply_enrichments(
        self,
        existing: list[AIComponent],
        agent_output: str,
    ) -> list[AIComponent]:
        """Merge enrichments from a JSON string."""
        data = self._parse_json(agent_output)
        if data is None:
            return list(existing)
        return self.apply_enrichments_from_dict(existing, data)

    def hydrate_component(self, component: AIComponent) -> AIComponent:
        """Optionally attach a code snippet to an existing component annotation."""
        if not self.include_code_snippets or component.decision_annotation is None:
            return component
        annotation = self._hydrate_code_snippet(
            component.decision_annotation,
            fallback_file_path=component.file_path,
            fallback_line_number=component.line_number,
        )
        return component.model_copy(update={"decision_annotation": annotation})

    def apply_enrichments_from_dict(
        self,
        existing: list[AIComponent],
        data: dict[str, Any],
    ) -> list[AIComponent]:
        """Merge enrichments, removals, and reclassifications into *existing*.

        Processing order:
        1. Remove components flagged by ``remove_components``.
        2. Reclassify components flagged by ``reclassify_components``.
        3. Apply field updates from ``enriched_components``.

        Returns a new list.  Components not referenced are passed through.

        Verdict scope
        -------------
        ``enriched_components`` and ``reclassify_components`` verdicts
        whose ``instance_id`` is not present in *existing* are dropped
        with a warning. They mutate per-component fields (``name``,
        ``component_type``, ``metadata``) and have no safe in-batch
        sibling to fall back on.

        ``remove_components`` is treated more leniently. The agent
        sometimes invents a line number or picks an out-of-batch sibling
        when expressing "this candidate is not a real AI component".
        Dropping those silently is a correctness bug because the
        downstream consolidation key fanout never gets a chance to run.
        Instead, when ``instance_id`` is unknown to the current batch
        we parse the ``name`` prefix from the iid (see
        :func:`_parse_iid_name_prefix`) and look for an in-batch
        sibling whose canonical name matches; if exactly one matches we
        redirect the removal to that sibling's consolidation key. The
        scan-pipeline-level :func:`_propagate_removals` then fans the
        decision out to every other instance sharing the same
        ``(name, type)`` key. Truly hallucinated iids (no in-batch
        sibling matches the parsed name) are still dropped with a
        warning.
        """

        batch_ids: set[str] = {c.instance_id for c in existing if c.instance_id}
        by_id: dict[str, AIComponent] = {
            c.instance_id: c for c in existing if c.instance_id
        }

        remove_ids: set[str] = set()
        remove_keys: set[tuple[str, str]] = set()
        for item in data.get("remove_components", []):
            iid = item.get("instance_id", "")
            if not iid:
                continue
            reason_text = item.get("reason", "")
            if iid in batch_ids:
                candidate = by_id.get(iid)
                if candidate is not None:
                    protect, why = _should_protect_deterministic_model_removal(
                        candidate, reason_text
                    )
                    if protect:
                        _LOGGER.warning(
                            "Rejected agent removal of deterministic model %s: "
                            "%s (agent reason: %s)",
                            iid, why, reason_text,
                        )
                        continue
                remove_ids.add(iid)
                _LOGGER.info(
                    "Agent removed component %s: %s",
                    iid, reason_text,
                )
                continue

            name_prefix = _parse_iid_name_prefix(iid)
            if not name_prefix:
                _LOGGER.warning(
                    "Dropping unparseable out-of-batch remove for %s "
                    "(reason: %s)",
                    iid, reason_text,
                )
                continue
            wanted = name_prefix.lower().strip()
            matches = [
                c for c in existing
                if (c.model_name or c.name).lower().strip() == wanted
            ]
            if not matches:
                _LOGGER.warning(
                    "Dropping out-of-batch remove for %s: no in-batch "
                    "sibling has canonical name '%s' (reason: %s)",
                    iid, wanted, reason_text,
                )
                continue
            if len({_ckey(c) for c in matches}) > 1:
                _LOGGER.warning(
                    "Dropping out-of-batch remove for %s: parsed name '%s' "
                    "is ambiguous across types %s (reason: %s)",
                    iid, wanted,
                    sorted({c.component_type.value for c in matches}),
                    reason_text,
                )
                continue
            sibling = matches[0]
            protect, why = _should_protect_deterministic_model_removal(
                sibling, reason_text
            )
            if protect:
                _LOGGER.warning(
                    "Rejected redirected removal of deterministic model %s "
                    "(out-of-batch iid %s): %s (agent reason: %s)",
                    sibling.instance_id, iid, why, reason_text,
                )
                continue
            ck = _ckey(sibling)
            remove_keys.add(ck)
            _LOGGER.info(
                "Agent remove redirected via consolidation key: "
                "out-of-batch iid %s → in-batch sibling %s "
                "(consolidation_key=%s, reason: %s)",
                iid, sibling.instance_id, ck, reason_text,
            )

        reclassify_map: dict[str, str] = {}
        for item in data.get("reclassify_components", []):
            iid = item.get("instance_id", "")
            new_type = item.get("new_type", "")
            if not (iid and new_type):
                continue
            if iid not in batch_ids:
                _LOGGER.warning(
                    "Dropping out-of-batch reclassify for %s → %s: "
                    "not in enrich_these (reason: %s)",
                    iid, new_type, item.get("reason", ""),
                )
                continue
            if new_type in _AGENT_CLASSIFICATION_TYPE_STRINGS:
                ok, reason = _verify_agent_evidence(
                    item.get("agent_evidence"),
                    allowed_roots=self.allowed_roots,
                )
                if not ok:
                    _LOGGER.warning(
                        "Rejected reclassify %s → %s: %s",
                        iid, new_type, reason,
                    )
                    continue
            reclassify_map[iid] = new_type
            _LOGGER.info(
                "Agent reclassified %s → %s: %s",
                iid, new_type, item.get("reason", ""),
            )

        reclassify_evidence: dict[str, dict[str, Any]] = {}
        for item in data.get("reclassify_components", []):
            iid = item.get("instance_id", "")
            if iid in reclassify_map:
                evidence = item.get("agent_evidence")
                if isinstance(evidence, dict):
                    reclassify_evidence[iid] = evidence

        updates_by_id: dict[str, dict[str, Any]] = {}
        enrichment_evidence: dict[str, dict[str, Any]] = {}
        annotations_by_id: dict[str, DecisionAnnotation] = {}
        for item in data.get("enriched_components", []):
            iid = item.get("instance_id", "")
            if not iid:
                continue
            if iid not in batch_ids:
                _LOGGER.warning(
                    "Dropping out-of-batch enrichment for %s: not in enrich_these",
                    iid,
                )
                continue
            raw_updates = item.get("updates", {}) or {}
            if isinstance(raw_updates, dict):
                raw_updates = dict(raw_updates)
                proposed_type = raw_updates.get("component_type")
                if (
                    isinstance(proposed_type, str)
                    and proposed_type in _AGENT_CLASSIFICATION_TYPE_STRINGS
                ):
                    ok, reason = _verify_agent_evidence(
                        item.get("agent_evidence"),
                        allowed_roots=self.allowed_roots,
                    )
                    if not ok:
                        _LOGGER.warning(
                            "Rejected enrichment %s → %s: %s (keeping other updates)",
                            iid, proposed_type, reason,
                        )
                        raw_updates.pop("component_type", None)
                    else:
                        evidence = item.get("agent_evidence")
                        if isinstance(evidence, dict):
                            enrichment_evidence[iid] = evidence
            updates_by_id[iid] = raw_updates
            annotation = self._decision_annotation_from_item(
                item,
                fallback_file_path=item.get("file_path", ""),
                fallback_line_number=item.get("line_number", 0),
            )
            if annotation is not None:
                annotations_by_id[iid] = annotation

        result: list[AIComponent] = []
        for comp in existing:
            if comp.instance_id in remove_ids:
                continue
            if remove_keys and _ckey(comp) in remove_keys:
                _LOGGER.info(
                    "Removing %s via consolidation-key fallback", comp.instance_id,
                )
                continue

            new_type_str = reclassify_map.get(comp.instance_id)
            if new_type_str:
                try:
                    new_type = AIComponentType(new_type_str)
                    merged_meta = dict(comp.metadata)
                    reclass_evidence = reclassify_evidence.get(comp.instance_id)
                    if reclass_evidence is not None:
                        merged_meta["agent_evidence"] = reclass_evidence
                    comp = comp.model_copy(update={
                        "component_type": new_type,
                        "needs_agentic": False,
                        "agentic_confidence": 0.8,
                        "metadata": merged_meta,
                    })
                except ValueError:
                    _LOGGER.warning(
                        "Invalid reclassify type '%s' for %s",
                        new_type_str, comp.instance_id,
                    )

            upd = updates_by_id.get(comp.instance_id)
            annotation = annotations_by_id.get(comp.instance_id)
            if upd is not None:
                merged_meta = dict(comp.metadata)
                incoming_meta = upd.pop("metadata", {})
                sanitized_incoming = _sanitize_metadata(
                    incoming_meta,
                    component_name=comp.name,
                    component_type=comp.component_type.value,
                )
                merged_meta.update(sanitized_incoming)
                raw_type = upd.pop("component_type", None)
                if isinstance(raw_type, str):
                    try:
                        upd["component_type"] = AIComponentType(raw_type)
                    except ValueError:
                        _LOGGER.warning("Invalid component_type '%s' in enrichment for %s", raw_type, comp.instance_id)
                enrich_evidence = enrichment_evidence.get(comp.instance_id)
                if enrich_evidence is not None:
                    merged_meta["agent_evidence"] = enrich_evidence
                comp = comp.model_copy(update={
                    **upd,
                    "metadata": merged_meta,
                    "decision_annotation": annotation,
                    "needs_agentic": False,
                    "agentic_confidence": 0.8,
                })
            elif comp.needs_agentic:
                update: dict[str, Any] = {"needs_agentic": False, "agentic_confidence": 0.8}
                if annotation is not None:
                    update["decision_annotation"] = annotation
                comp = comp.model_copy(update=update)

            if comp.component_type == AIComponentType.MODEL:
                effective_name = comp.model_name or comp.name
                if _is_class_name_not_model_id(effective_name):
                    _LOGGER.warning(
                        "Removing model component '%s': class name '%s' is not a model ID",
                        comp.instance_id,
                        effective_name,
                    )
                    continue

            comp = _normalize_endpoint_model_name(comp)
            comp = _rewrite_if_ungrounded_endpoint(
                comp,
                allowed_roots=self.allowed_roots,
            )
            comp = _cap_confidence_if_unresolved(comp)
            result.append(comp)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        """Parse the agent's final message as JSON."""
        text = text.strip()
        if not text:
            _LOGGER.warning("Agent returned empty output")
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            _LOGGER.warning(
                "Failed to parse agent JSON output — first 300 chars: %s",
                text[:300],
            )
            return None

    def _extract_new_components(self, data: dict[str, Any]) -> list[AIComponent]:
        components: list[AIComponent] = []
        for item in data.get("new_components", []):
            try:
                comp_type = AIComponentType(item.get("component_type", "other"))
            except ValueError:
                comp_type = AIComponentType.OTHER
            name = item.get("name", "unknown")
            if _is_sentinel_name(name):
                _LOGGER.warning(
                    "Rejected new %s component: sentinel/placeholder name %r — "
                    "LLM should omit the entry instead",
                    comp_type.value, name,
                )
                continue
            decision = item.get("decision_annotation") or {}
            justification = decision.get("justification", "") if isinstance(decision, dict) else ""
            if _is_negating_justification(justification):
                _LOGGER.warning(
                    "Rejected new %s component '%s': self-contradicting "
                    "justification %r — LLM decided not to add it",
                    comp_type.value, name, justification,
                )
                continue
            if comp_type == AIComponentType.MODEL and _is_class_name_not_model_id(name):
                _LOGGER.warning(
                    "Rejected new model component '%s': class name, not a model ID",
                    name,
                )
                continue
            if comp_type in _AGENT_CLASSIFICATION_TYPES:
                ok, reason = _verify_agent_evidence(
                    item.get("agent_evidence"),
                    allowed_roots=self.allowed_roots,
                )
                if not ok:
                    _LOGGER.warning(
                        "Rejected new %s component '%s': %s",
                        comp_type.value, name, reason,
                    )
                    continue
            sanitized_meta = _sanitize_metadata(
                item.get("metadata", {}),
                component_name=name,
                component_type=comp_type.value,
            )
            probe = AIComponent(
                name=name,
                component_type=comp_type,
                file_path=item.get("file_path", ""),
                framework=item.get("framework", ""),
                metadata=sanitized_meta,
            )
            tool_reject, tool_reason = _should_reject_tool_from_helm(probe)
            if tool_reject:
                _LOGGER.warning(
                    "Rejected new tool component '%s': %s "
                    "(Helm/K8s service, not an agent-callable tool)",
                    name, tool_reason,
                )
                continue
            new_comp = AIComponent(
                name=name,
                component_type=comp_type,
                file_path=item.get("file_path", ""),
                line_number=item.get("line_number", 0),
                framework=item.get("framework", ""),
                model_name=item.get("model_name"),
                detection_source=DetectionSource.AGENTIC,
                decision_annotation=self._decision_annotation_from_item(
                    item,
                    fallback_file_path=item.get("file_path", ""),
                    fallback_line_number=item.get("line_number", 0),
                ),
                metadata=sanitized_meta,
            )
            new_comp = _normalize_endpoint_model_name(new_comp)
            new_comp = _rewrite_if_ungrounded_endpoint(
                new_comp,
                allowed_roots=self.allowed_roots,
            )
            new_comp = _cap_confidence_if_unresolved(new_comp)
            components.append(new_comp)
        return components

    @staticmethod
    def _extract_enrichments(data: dict[str, Any]) -> list[AIComponent]:
        """Enrichments don't create new components; they update existing ones.

        We return an empty list here -- actual merging is done via
        :meth:`apply_enrichments`.
        """
        return []

    def _extract_relationships(self, data: dict[str, Any]) -> list[ComponentRelationship]:
        relationships: list[ComponentRelationship] = []
        for item in data.get("new_relationships", []):
            try:
                rel_type = RelationshipType(item.get("relationship_type", "CUSTOM"))
            except ValueError:
                rel_type = RelationshipType.CUSTOM
            source_name = item.get("source_name", "")
            target_name = item.get("target_name", "")
            if _is_sentinel_name(source_name) or _is_sentinel_name(target_name):
                _LOGGER.warning(
                    "Rejected new %s relationship: sentinel/placeholder endpoint "
                    "(source=%r target=%r) — LLM should omit the entry instead",
                    rel_type.value, source_name, target_name,
                )
                continue
            decision = item.get("decision_annotation") or {}
            justification = decision.get("justification", "") if isinstance(decision, dict) else ""
            if _is_negating_justification(justification):
                _LOGGER.warning(
                    "Rejected new %s relationship '%s -> %s': self-contradicting "
                    "justification %r — LLM decided not to add it",
                    rel_type.value, source_name, target_name, justification,
                )
                continue
            src_type = AIComponentType.OTHER
            tgt_type = AIComponentType.OTHER
            if item.get("source_type"):
                try:
                    src_type = AIComponentType(item["source_type"])
                except ValueError:
                    pass
            if item.get("target_type"):
                try:
                    tgt_type = AIComponentType(item["target_type"])
                except ValueError:
                    pass
            relationships.append(
                ComponentRelationship(
                    source_instance_id="",
                    target_instance_id="",
                    source_name=source_name,
                    target_name=target_name,
                    source_type=src_type,
                    target_type=tgt_type,
                    relationship_type=rel_type,
                    decision_annotation=self._decision_annotation_from_item(item),
                )
            )
        return relationships

    @staticmethod
    def _drop_dependency_mcp_client_rels(
        relationships: list[ComponentRelationship],
    ) -> list[ComponentRelationship]:
        """Drop USES_MCP_CLIENT where target is a package dependency, not a real client."""
        result: list[ComponentRelationship] = []
        for rel in relationships:
            if (
                rel.relationship_type == RelationshipType.USES_MCP_CLIENT
                and rel.target_type == AIComponentType.DEPENDENCY
            ):
                _LOGGER.debug(
                    "Dropped dependency-typed USES_MCP_CLIENT: %s -> %s",
                    rel.source_name, rel.target_name,
                )
                continue
            result.append(rel)
        return result

    @staticmethod
    def _drop_dep_to_dep_uses_model(
        relationships: list[ComponentRelationship],
    ) -> list[ComponentRelationship]:
        """Drop USES_MODEL where both source and target are package dependencies."""
        result: list[ComponentRelationship] = []
        for rel in relationships:
            if (
                rel.relationship_type == RelationshipType.USES_MODEL
                and rel.source_type == AIComponentType.DEPENDENCY
                and rel.target_type == AIComponentType.DEPENDENCY
            ):
                _LOGGER.debug(
                    "Dropped dep-to-dep USES_MODEL: %s -> %s",
                    rel.source_name, rel.target_name,
                )
                continue
            result.append(rel)
        return result

    def _extract_risk_flags(self, data: dict[str, Any]) -> list[RiskFlag]:
        from ..risk import RISK_WEIGHTS

        flags: list[RiskFlag] = []
        for item in data.get("risk_findings", []):
            flag_name = item.get("flag", "")
            try:
                severity = Severity(item.get("severity", "info"))
            except ValueError:
                severity = Severity.INFO

            weight_info = RISK_WEIGHTS.get(flag_name, {})
            weight = weight_info.get("weight", 5)

            flags.append(
                RiskFlag(
                    flag=flag_name,
                    severity=severity,
                    weight=weight,
                    description=item.get("description", ""),
                    file_path=item.get("file_path", ""),
                    line_number=item.get("line_number", 0),
                    decision_annotation=self._decision_annotation_from_item(
                        item,
                        fallback_file_path=item.get("file_path", ""),
                        fallback_line_number=item.get("line_number", 0),
                    ),
                )
            )
        return flags

    @staticmethod
    def _enforce_embedding_resolution(
        components: list[AIComponent],
        relationships: list[ComponentRelationship],
    ) -> list[AIComponent]:
        """Remove embedding components that lack a concrete model identifier.

        An embedding component is kept only if:
        - ``embedding_model`` is set and non-empty, or
        - ``model_name`` is set and is not a class name, or
        - a ``USES_EMBEDDING`` relationship exists targeting a concrete model.
        """
        has_embedding_rel: set[str] = set()
        for rel in relationships:
            if rel.relationship_type == RelationshipType.USES_EMBEDDING:
                has_embedding_rel.add(rel.source_name)
                has_embedding_rel.add(rel.source_instance_id)

        kept: list[AIComponent] = []
        for comp in components:
            if comp.component_type != AIComponentType.EMBEDDING:
                kept.append(comp)
                continue
            if comp.embedding_model:
                kept.append(comp)
                continue
            mn = comp.model_name or ""
            if mn and not _is_class_name_not_model_id(mn):
                kept.append(comp)
                continue
            if comp.name in has_embedding_rel or comp.instance_id in has_embedding_rel:
                kept.append(comp)
                continue
            _LOGGER.warning(
                "Removing unresolved embedding wrapper '%s' (%s): "
                "no concrete model identifier",
                comp.name,
                comp.instance_id,
            )
        return kept

    def _decision_annotation_from_item(
        self,
        item: dict[str, Any],
        *,
        fallback_file_path: str = "",
        fallback_line_number: int = 0,
    ) -> DecisionAnnotation | None:
        raw = item.get("decision_annotation")
        if not raw:
            return None
        try:
            annotation = DecisionAnnotation.model_validate(raw)
        except ValueError:
            _LOGGER.warning("Invalid decision_annotation in agent output: %s", raw)
            return None
        if not self.include_code_snippets:
            return annotation
        return self._hydrate_code_snippet(
            annotation,
            fallback_file_path=fallback_file_path,
            fallback_line_number=fallback_line_number,
        )

    def _hydrate_code_snippet(
        self,
        annotation: DecisionAnnotation,
        *,
        fallback_file_path: str = "",
        fallback_line_number: int = 0,
    ) -> DecisionAnnotation:
        if annotation.code_snippet is not None:
            return annotation

        location = next(
            (
                loc for loc in annotation.evidence_locations
                if loc.file_path and loc.start_line > 0
            ),
            None,
        )
        if location is None and fallback_file_path and fallback_line_number > 0:
            location = EvidenceLocation(
                file_path=fallback_file_path,
                start_line=fallback_line_number,
                end_line=fallback_line_number,
                role="primary",
            )
        if location is None:
            return annotation

        snippet = self._read_code_snippet(
            location.file_path,
            location.start_line,
            location.end_line or location.start_line,
        )
        if snippet is None:
            return annotation
        return annotation.model_copy(update={"code_snippet": snippet})

    def _read_code_snippet(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        *,
        max_lines: int = 30,
    ) -> CodeSnippet | None:
        if not file_path or start_line <= 0:
            return None
        if self.allowed_roots:
            try:
                resolved = Path(file_path).resolve()
            except OSError:
                return None
            if not any(
                resolved == Path(r).resolve() or Path(r).resolve() in resolved.parents
                for r in self.allowed_roots
            ):
                return None
        path = Path(file_path)
        try:
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        except OSError:
            return None
        if not lines or start_line > len(lines):
            return None

        bounded_end = max(start_line, end_line)
        excerpt_end = min(bounded_end, start_line + max_lines - 1, len(lines))
        excerpt = "".join(lines[start_line - 1:excerpt_end])
        return CodeSnippet(
            file_path=file_path,
            start_line=start_line,
            end_line=excerpt_end,
            text=excerpt,
            truncated=bounded_end > excerpt_end,
        )
