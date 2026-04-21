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

"""Offline A2A Agent Card detector.

This scanner is strictly offline: it never performs network calls. It finds
A2A (Agent-to-Agent, https://a2a-protocol.org) agent descriptions by
parsing what is already on disk:

1. **JSON Agent Cards.** Any JSON file on the filesystem that matches the
   Agent Card schema — strongly when the filename is ``agent-card.json``
   or ``agent.json`` under a ``.well-known/`` path segment, and
   conservatively shape-matched for arbitrarily-placed JSON.
2. **Python-embedded Agent Cards.** ``AgentCard(...)``, ``A2AServer(...)``,
   and ``A2AClient(...)`` constructor calls detected via the existing
   libcst pipeline (``aibom.cst_parser.parse_source_code``). Keyword
   arguments are already extracted as structured values by the parser,
   so nested ``skills=[{...}]`` or ``agent_card={...}`` payloads are
   available without any extra regex work.

Both sources produce :class:`AIComponent` entries of type ``AGENT`` whose
``metadata["agent_card"]`` is a normalized, redacted view of the card
(name, description, version, skills, endpoints, capabilities). This
metadata is the ground truth that the Phase-4 remote-agent resolver and
the Phase-5 LLM verification gate consume to decide whether a remote
proxy is an agent.

The detector never stores the contents of ``securitySchemes`` or
``signatures`` verbatim — only types/ids/presence flags — to avoid
leaking credentials or JWKS/JWS payloads into the AIBOM.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from pathspec import PathSpec

from ..cst_parser import parse_source_code
from ..models import AIComponent, ComponentRelationship, ScanContext
from ..models.enums import AIComponentType, DetectionSource, RelationshipType
from ..structures import CallObservation, CodeAnalysisResult
from .base import BaseScanner
from .file_cache import is_python_source, read_python_source, read_text_cached

_LOGGER = logging.getLogger(__name__)

_WELL_KNOWN_FILENAMES: frozenset[str] = frozenset({
    "agent.json",
    "agent-card.json",
})

_WELL_KNOWN_SEGMENT = ".well-known"

_MAX_JSON_SIZE_BYTES = 512 * 1024

_PY_SUBSTR_HINTS: tuple[str, ...] = (
    "a2a",
    "AgentCard",
    "A2AServer",
    "A2AClient",
    "agent_card",
    ".well-known/agent",
)

_AGENT_CARD_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "name": ("name",),
    "description": ("description",),
    "version": ("version",),
    "provider": ("provider",),
    "documentation_url": ("documentationUrl", "documentation_url"),
    "icon_url": ("iconUrl", "icon_url"),
    "capabilities": ("capabilities",),
    "skills": ("skills",),
    "supported_interfaces": ("supportedInterfaces", "supported_interfaces"),
    "default_input_modes": ("defaultInputModes", "default_input_modes"),
    "default_output_modes": ("defaultOutputModes", "default_output_modes"),
    "security_schemes": ("securitySchemes", "security_schemes"),
    "security_requirements": (
        "securityRequirements",
        "security_requirements",
        "security",
    ),
    "signatures": ("signatures",),
    "api": ("api",),
    "auth": ("auth",),
    "identity": ("identity",),
    "pricing": ("pricing",),
    "protocol_version": ("protocolVersion", "protocol_version"),
}

_AGENT_CARD_SHAPE_KEYS: frozenset[str] = frozenset(
    alias for aliases in _AGENT_CARD_FIELD_ALIASES.values() for alias in aliases
)

_SKILL_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id",),
    "name": ("name",),
    "description": ("description",),
    "tags": ("tags",),
    "examples": ("examples",),
    "input_modes": ("inputModes", "input_modes"),
    "output_modes": ("outputModes", "output_modes"),
}

_INTERFACE_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "url": ("url",),
    "protocol_binding": ("protocolBinding", "protocol_binding"),
    "protocol_version": ("protocolVersion", "protocol_version"),
    "type": ("type",),
}


def _load_exclude_spec(patterns: list[str]) -> Optional[PathSpec]:
    if not patterns:
        return None
    return PathSpec.from_lines("gitwildmatch", patterns)


def _is_excluded(
    file_path: Path, root: Path, spec: Optional[PathSpec]
) -> bool:
    if not spec:
        return False
    try:
        rel = file_path.relative_to(root).as_posix()
    except ValueError:
        rel = file_path.as_posix()
    return spec.match_file(rel)


def _iter_files_under(root: Path, spec: Optional[PathSpec]) -> list[Path]:
    out: list[Path] = []
    if root.is_file():
        if not _is_excluded(root, root.parent, spec):
            out.append(root)
        return out
    if not root.is_dir():
        return out
    for p in root.rglob("*"):
        if p.is_file() and not _is_excluded(p, root, spec):
            out.append(p)
    return out


def _is_well_known_agent_card_path(path: Path) -> bool:
    """Return True when *path* is the canonical A2A Agent Card location.

    Matches ``<anything>/.well-known/agent.json`` and
    ``<anything>/.well-known/agent-card.json``. The ``.well-known``
    segment must appear as a parent directory, not merely be a substring
    of some other component — so ``well-known-thing/agent.json`` will
    NOT match, and neither will ``agent.json`` in a random location.
    """
    if path.name not in _WELL_KNOWN_FILENAMES:
        return False
    return _WELL_KNOWN_SEGMENT in path.parts[:-1]


def _lookup_alias(data: dict[str, Any], canonical_key: str) -> Any:
    """Return the first matching aliased value for *canonical_key*."""
    for alias in _AGENT_CARD_FIELD_ALIASES.get(canonical_key, (canonical_key,)):
        if alias in data:
            return data[alias]
    return None


def _looks_like_agent_card_shape(data: Any) -> tuple[bool, str]:
    """Return ``(is_agent_card, reason)`` for an arbitrary parsed JSON/dict.

    ``reason`` is a short debug string that names the strongest shape
    signal (well-known path, skills-array-shape, etc.). Callers use this
    in the BOM metadata so operators can see why a file was flagged.

    The heuristic is intentionally conservative to avoid false positives
    on unrelated JSON files:

    * Must be a non-empty dict at the top level.
    * Must contain a ``name`` field (string).
    * Must contain at least one of the A2A-specific structural keys:
      ``skills`` (as a list of dicts with ``id`` or ``name``),
      ``supportedInterfaces`` / ``supported_interfaces``,
      ``capabilities`` (as a dict — not an array of bare strings, which
      are a common tsconfig shape), ``api``, or ``identity`` (legacy
      schema variant).
    * ``skills`` alone is not enough unless the elements look like A2A
      skill objects (dicts with at least an ``id`` or ``name``).
    """
    if not isinstance(data, dict) or not data:
        return False, ""

    if "name" not in data or not isinstance(data["name"], str):
        return False, ""

    skills = _lookup_alias(data, "skills")
    if isinstance(skills, list) and skills:
        first = skills[0]
        if isinstance(first, dict) and ("id" in first or "name" in first):
            return True, "skills_array"

    supported = _lookup_alias(data, "supported_interfaces")
    if isinstance(supported, list) and supported:
        first = supported[0]
        if isinstance(first, dict) and (
            "url" in first
            or "protocolBinding" in first
            or "protocol_binding" in first
        ):
            return True, "supported_interfaces"

    api = _lookup_alias(data, "api")
    if isinstance(api, dict) and isinstance(api.get("type"), str) and api.get("type").lower() == "a2a":
        return True, "api_type_a2a"

    capabilities = _lookup_alias(data, "capabilities")
    if isinstance(capabilities, dict):
        capability_signals = {
            "streaming",
            "pushNotifications",
            "stateTransitionHistory",
            "extendedAgentCard",
        }
        if capability_signals & set(capabilities.keys()):
            return True, "capabilities_dict"

    identity = _lookup_alias(data, "identity")
    if isinstance(identity, dict) and isinstance(identity.get("name"), str):
        if isinstance(data.get("skills"), list) or isinstance(
            data.get("capabilities"), (list, dict)
        ):
            return True, "identity_legacy_schema"

    return False, ""


def _summarize_skill(raw: Any) -> Optional[dict[str, Any]]:
    """Return a normalized skill dict or ``None`` if *raw* is malformed."""
    if not isinstance(raw, dict):
        return None
    out: dict[str, Any] = {}
    for canonical, aliases in _SKILL_FIELD_ALIASES.items():
        for alias in aliases:
            if alias in raw:
                out[canonical] = raw[alias]
                break
    if not out.get("id") and not out.get("name"):
        return None
    return out


def _summarize_interface(raw: Any) -> Optional[dict[str, Any]]:
    """Return a normalized interface dict or ``None`` if *raw* is malformed."""
    if not isinstance(raw, dict):
        return None
    out: dict[str, Any] = {}
    for canonical, aliases in _INTERFACE_FIELD_ALIASES.items():
        for alias in aliases:
            if alias in raw:
                out[canonical] = raw[alias]
                break
    if not (out.get("url") or out.get("protocol_binding") or out.get("type")):
        return None
    return out


def _redacted_security_summary(raw: Any) -> dict[str, Any]:
    """Summarize security_schemes WITHOUT including URLs, secrets, or keys.

    Only scheme type labels and the list of registered ids are recorded.
    """
    summary: dict[str, Any] = {}
    if not isinstance(raw, dict):
        return summary
    ids_by_type: dict[str, list[str]] = {}
    for scheme_id, scheme in raw.items():
        scheme_type = "unknown"
        if isinstance(scheme, dict):
            nested_keys = set(scheme.keys())
            if "type" in nested_keys and isinstance(scheme["type"], str):
                scheme_type = scheme["type"]
            else:
                for candidate in (
                    "openIdConnectSecurityScheme",
                    "httpSecurityScheme",
                    "apiKeySecurityScheme",
                    "oauth2SecurityScheme",
                ):
                    if candidate in nested_keys:
                        scheme_type = candidate.replace("SecurityScheme", "")
                        break
        ids_by_type.setdefault(scheme_type, []).append(str(scheme_id))
    if ids_by_type:
        summary["scheme_ids_by_type"] = {
            k: sorted(v) for k, v in ids_by_type.items()
        }
    return summary


def _extract_endpoints(normalized: dict[str, Any]) -> list[str]:
    """Flatten all endpoint URLs from supportedInterfaces and api."""
    urls: list[str] = []
    for iface in normalized.get("supported_interfaces") or []:
        if isinstance(iface, dict):
            url = iface.get("url")
            if isinstance(url, str) and url.strip():
                urls.append(url.strip())
    api = normalized.get("api")
    if isinstance(api, dict):
        url = api.get("url")
        if isinstance(url, str) and url.strip():
            urls.append(url.strip())
    seen: set[str] = set()
    deduped: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _normalize_agent_card(data: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize an Agent Card into a stable, snake_case, redacted view."""
    if not isinstance(data, dict):
        return {}

    card: dict[str, Any] = {}
    for canonical in ("name", "description", "version"):
        value = _lookup_alias(data, canonical)
        if isinstance(value, str) and value.strip():
            card[canonical] = value.strip()

    provider = _lookup_alias(data, "provider")
    if isinstance(provider, dict):
        provider_summary: dict[str, Any] = {}
        for field in ("organization", "url", "name"):
            if field in provider and isinstance(provider[field], str):
                provider_summary[field] = provider[field]
        if provider_summary:
            card["provider"] = provider_summary

    for canonical in ("documentation_url", "icon_url"):
        value = _lookup_alias(data, canonical)
        if isinstance(value, str) and value.strip():
            card[canonical] = value.strip()

    capabilities = _lookup_alias(data, "capabilities")
    if isinstance(capabilities, dict):
        card["capabilities"] = {
            k: bool(v) for k, v in capabilities.items() if isinstance(v, (bool, int))
        }
    elif isinstance(capabilities, list):
        card["capabilities_list"] = [
            str(item) for item in capabilities if isinstance(item, (str, int))
        ]

    skills_raw = _lookup_alias(data, "skills")
    if isinstance(skills_raw, list):
        summarized = [
            s for s in (_summarize_skill(raw) for raw in skills_raw) if s is not None
        ]
        if summarized:
            card["skills"] = summarized

    interfaces_raw = _lookup_alias(data, "supported_interfaces")
    if isinstance(interfaces_raw, list):
        summarized_ifaces = [
            s
            for s in (_summarize_interface(raw) for raw in interfaces_raw)
            if s is not None
        ]
        if summarized_ifaces:
            card["supported_interfaces"] = summarized_ifaces

    api = _lookup_alias(data, "api")
    if isinstance(api, dict):
        api_summary: dict[str, Any] = {}
        for field in ("type", "url", "protocol_version"):
            value = api.get(field) or api.get(
                "protocolVersion" if field == "protocol_version" else field
            )
            if isinstance(value, str):
                api_summary[field] = value
        if api_summary:
            card["api"] = api_summary

    for canonical in ("default_input_modes", "default_output_modes"):
        modes = _lookup_alias(data, canonical)
        if isinstance(modes, list):
            filtered = [m for m in modes if isinstance(m, str)]
            if filtered:
                card[canonical] = filtered

    security = _redacted_security_summary(_lookup_alias(data, "security_schemes"))
    if security:
        card["security_schemes"] = security

    signatures = _lookup_alias(data, "signatures")
    if isinstance(signatures, list) and signatures:
        card["signatures_present"] = True
        card["signatures_count"] = len(signatures)

    endpoints = _extract_endpoints(card)
    if endpoints:
        card["endpoints"] = endpoints

    return card


def _pick_component_name(card: dict[str, Any], fallback: str) -> str:
    name = card.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    provider = card.get("provider")
    if isinstance(provider, dict):
        org = provider.get("organization") or provider.get("name")
        if isinstance(org, str) and org.strip():
            return f"{org.strip()} A2A Agent"
    return fallback


def _parse_json_file(path: Path) -> Optional[Any]:
    """Load and JSON-parse *path*. Returns ``None`` on any error."""
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size > _MAX_JSON_SIZE_BYTES:
        return None
    try:
        text = read_text_cached(path)
    except OSError:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _component_from_json_agent_card(
    path: Path, data: dict[str, Any], shape_reason: str
) -> Optional[AIComponent]:
    card = _normalize_agent_card(data)
    if not card:
        return None
    fallback_name = path.stem or "a2a_agent"
    component_name = _pick_component_name(card, fallback_name)
    well_known = _is_well_known_agent_card_path(path)
    metadata: dict[str, Any] = {
        "agent_card": card,
        "source_shape": "json",
        "well_known": well_known,
        "shape_reason": shape_reason,
    }
    return AIComponent(
        name=component_name,
        component_type=AIComponentType.AGENT,
        file_path=str(path.resolve()),
        line_number=0,
        framework="a2a",
        detection_source=DetectionSource.CONFIG_FILE,
        heuristic_confidence=1.0,
        needs_agentic=True,
        agentic_hint=(
            f"A2A Agent Card JSON ({shape_reason}); confirm skills and "
            "endpoints are intentional."
        ),
        metadata=metadata,
    )


def _scan_json_files(paths: list[Path]) -> list[AIComponent]:
    components: list[AIComponent] = []
    for path in paths:
        if path.suffix.lower() != ".json":
            continue
        well_known = _is_well_known_agent_card_path(path)
        data = _parse_json_file(path)
        if data is None:
            continue
        if not isinstance(data, dict):
            continue
        shape_is_card, shape_reason = _looks_like_agent_card_shape(data)
        if not shape_is_card and not well_known:
            continue
        if well_known and not shape_reason:
            shape_reason = "well_known_path"
        comp = _component_from_json_agent_card(path, data, shape_reason)
        if comp is not None:
            components.append(comp)
    return components


def _py_text_preselect(text: str) -> bool:
    """Cheap substring filter: skip CST parsing for files with no A2A hints."""
    return any(hint in text for hint in _PY_SUBSTR_HINTS)


def _match_agent_card_call_name(qn: str) -> bool:
    """Match constructors that indicate an Agent Card payload."""
    tail = qn.rsplit(".", 1)[-1] if qn else ""
    if tail == "AgentCard":
        return True
    return qn.endswith(".AgentCard") or qn == "AgentCard"


def _match_a2a_server_call_name(qn: str) -> bool:
    tail = qn.rsplit(".", 1)[-1] if qn else ""
    return tail in {"A2AServer", "A2AStarletteApplication", "A2AFastAPIApplication"}


def _match_a2a_client_call_name(qn: str) -> bool:
    tail = qn.rsplit(".", 1)[-1] if qn else ""
    return tail in {"A2AClient", "A2ACardResolver"}


def _kwargs_to_agent_card_dict(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Treat a constructor's kwargs as an Agent Card payload.

    Values stored as ``VARIABLE:...`` or ``ATTRIBUTE:...`` (symbols whose
    literal value the CST parser could not resolve) are dropped — they'd
    pollute the BOM without adding information.
    """
    card: dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, str) and (
            value.startswith("VARIABLE:")
            or value.startswith("ATTRIBUTE:")
            or value.startswith("CALL:")
            or value.startswith("UNHANDLED_NODE:")
        ):
            continue
        card[key] = value
    return card


def _resolve_agent_card_kwarg(
    server_kwargs: dict[str, Any], result: CodeAnalysisResult
) -> tuple[dict[str, Any], Optional[int]]:
    """Resolve ``A2AServer(agent_card=my_card)`` to ``my_card``'s literal.

    Returns ``(card_payload, resolved_line)``. ``resolved_line`` is the
    line number of the ``AgentCard(...)`` assignment the server refers
    to — callers use it to suppress emitting the bare ``AgentCard(...)``
    as a separate component when the server is already emitting the same
    card.

    If ``agent_card`` is a bare name (``VARIABLE:my_card``), look up the
    matching assignment in the same file and, if it was an
    ``AgentCard(...)`` call, return that call's kwargs; otherwise fall
    back to the server's own kwargs. Dict-literal assignments (as in
    ``my_card = {"name": ...}``) are not tracked by the existing CST
    assignment observer, so the direct-kwarg dict form (``A2AServer(
    agent_card={...})``) remains the primary supported shape.
    """
    raw = server_kwargs.get("agent_card")
    if isinstance(raw, dict):
        return raw, None
    if isinstance(raw, str) and raw.startswith("VARIABLE:"):
        var_name = raw.removeprefix("VARIABLE:")
        for assignment in result.assignments:
            if assignment.target_qualified_name == var_name:
                if _match_agent_card_call_name(assignment.call.qualified_name):
                    return (
                        _kwargs_to_agent_card_dict(assignment.call.arguments),
                        assignment.call.line_number,
                    )
    return _kwargs_to_agent_card_dict(server_kwargs), None


def _component_from_python_call(
    path: Path,
    call: CallObservation,
    card_source: dict[str, Any],
    constructor_label: str,
    result: CodeAnalysisResult,
) -> Optional[AIComponent]:
    card = _normalize_agent_card(card_source)
    if not card:
        return None
    has_identity = bool(card.get("name") or card.get("description"))
    has_structural = bool(
        card.get("skills")
        or card.get("supported_interfaces")
        or card.get("api")
        or card.get("capabilities")
        or card.get("endpoints")
    )
    if not (has_identity or has_structural):
        return None
    fallback_name = (path.stem or "a2a_agent") + f"_{constructor_label.lower()}"
    component_name = _pick_component_name(card, fallback_name)
    metadata: dict[str, Any] = {
        "agent_card": card,
        "source_shape": "python",
        "constructor": constructor_label,
        "qualified_name": call.qualified_name,
    }
    _ = result
    return AIComponent(
        name=component_name,
        component_type=AIComponentType.AGENT,
        file_path=str(path.resolve()),
        line_number=call.line_number,
        framework="a2a",
        detection_source=DetectionSource.CODE_ANALYSIS,
        heuristic_confidence=1.0,
        needs_agentic=True,
        agentic_hint=(
            f"A2A {constructor_label} constructor with Agent Card payload; "
            "confirm this class actually orchestrates an agent loop."
        ),
        metadata=metadata,
    )


def _scan_python_file(path: Path) -> list[AIComponent]:
    try:
        text = read_python_source(path)
    except OSError:
        return []
    if not text or not _py_text_preselect(text):
        return []
    try:
        result = parse_source_code(str(path), text)
    except (ValueError, AttributeError):
        return []

    components: list[AIComponent] = []
    seen_positions: set[tuple[int, str]] = set()
    server_resolved_card_lines: set[int] = set()

    def _server_call(call: CallObservation) -> None:
        qn = call.qualified_name
        position_key = (call.line_number, qn)
        if position_key in seen_positions:
            return
        seen_positions.add(position_key)
        card_source, resolved_line = _resolve_agent_card_kwarg(
            call.arguments, result
        )
        if resolved_line is not None:
            server_resolved_card_lines.add(resolved_line)
        comp = _component_from_python_call(
            path, call, card_source, "A2AServer", result
        )
        if comp is not None:
            components.append(comp)

    def _card_call(call: CallObservation) -> None:
        qn = call.qualified_name
        position_key = (call.line_number, qn)
        if position_key in seen_positions:
            return
        if call.line_number in server_resolved_card_lines:
            return
        seen_positions.add(position_key)
        card_source = _kwargs_to_agent_card_dict(call.arguments)
        comp = _component_from_python_call(
            path, call, card_source, "AgentCard", result
        )
        if comp is not None:
            components.append(comp)

    for call in result.calls:
        if _match_a2a_server_call_name(call.qualified_name):
            _server_call(call)
    for assignment in result.assignments:
        if _match_a2a_server_call_name(assignment.call.qualified_name):
            _server_call(assignment.call)

    for call in result.calls:
        if _match_agent_card_call_name(call.qualified_name):
            _card_call(call)
    for assignment in result.assignments:
        if _match_agent_card_call_name(assignment.call.qualified_name):
            _card_call(assignment.call)

    return components


def iter_scanned_agent_cards(context: ScanContext) -> list[AIComponent]:
    """Return every A2A Agent Card found on disk for ``context``'s paths.

    Single source of truth for Agent Card discovery. Both
    :class:`A2ADetector` and the Phase-4 remote-agent resolver call this
    so the resolver can build a URL→card index without triggering a
    second full scan or re-emitting duplicate components.

    All returned :class:`AIComponent` entries have ``component_type =
    AGENT`` and ``framework = 'a2a'`` with ``metadata['agent_card']``
    populated by :func:`_normalize_agent_card`. Duplicates (same file,
    line, and card name) are collapsed.
    """
    all_files: list[Path] = []
    idx = context.file_index()
    if idx:
        all_files = [e.path for entries in idx.values() for e in entries]
    else:
        spec = _load_exclude_spec(context.exclude_patterns)
        for raw in context.paths:
            root = Path(raw).expanduser()
            all_files.extend(_iter_files_under(root, spec))

    components: list[AIComponent] = []

    json_files = [p for p in all_files if p.suffix.lower() == ".json"]
    components.extend(_scan_json_files(json_files))

    seen_keys: set[tuple[str, int, str]] = set()
    deduped: list[AIComponent] = []
    for comp in components:
        key = (comp.file_path, comp.line_number, comp.name)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(comp)
    components = deduped

    for path in all_files:
        if is_python_source(path):
            components.extend(_scan_python_file(path))

    final_seen: set[tuple[str, int, str]] = set()
    final: list[AIComponent] = []
    for comp in components:
        key = (comp.file_path, comp.line_number, comp.name)
        if key in final_seen:
            continue
        final_seen.add(key)
        final.append(comp)

    return final


class A2ADetector(BaseScanner):
    """Offline A2A Agent Card detector.

    Runs in the shared scanner pipeline. Produces
    :class:`AIComponent` entries of type ``AGENT`` whenever an Agent Card
    is found on disk (JSON or Python-embedded); never performs network
    calls. Relationships are not emitted here — the Phase-4 remote agent
    resolver owns ``INVOKES_A2A_AGENT`` / ``EXPOSES_A2A_AGENT`` wiring
    because it needs cross-file context that this detector does not try
    to reconstruct.
    """

    name = "a2a_detector"

    def supports(self, context: ScanContext) -> bool:
        return True

    def scan(
        self, context: ScanContext
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        components = iter_scanned_agent_cards(context)
        _LOGGER.debug(
            "A2ADetector detected %d Agent Card components", len(components)
        )
        _ = RelationshipType
        return components, []


__all__ = [
    "A2ADetector",
    "iter_scanned_agent_cards",
    "_is_well_known_agent_card_path",
    "_looks_like_agent_card_shape",
    "_normalize_agent_card",
    "_redacted_security_summary",
]
