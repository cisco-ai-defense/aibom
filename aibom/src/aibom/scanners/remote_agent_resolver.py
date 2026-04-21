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

"""Offline remote-agent resolver (A2A first, remote-agent-SDK fallback).

This scanner is strictly offline: it never performs network calls. It looks
for local code that *invokes* a remote agent (as opposed to exposing one —
that is the Phase-3 :mod:`aibom.scanners.a2a_detector`'s job) and emits an
:class:`AIComponent` of type
:data:`~aibom.models.enums.AIComponentType.AGENT_PROXY` for each such
invocation site.

A2A signals (preferred, higher specificity)
-------------------------------------------

Four offline signals are considered, in decreasing order of specificity:

1. **Client constructor calls** — ``A2AClient(url=…)``, ``A2ACardResolver(
   base_url=…)``, ``AgentClient(endpoint=…)``, detected via the existing
   libcst pipeline.
2. **Canonical Agent Card URL literals** — string literals ending in
   ``/.well-known/agent.json`` or ``/.well-known/agent-card.json``
   that appear inside a class body. The path is an A2A-spec artifact,
   so a literal alone is enough evidence to record a proxy. Module-
   scope literals are deliberately ignored (too noisy — see
   :mod:`aibom.cst_parser`'s ``protocol_strings`` semantics).
3. **Class inheritance from an A2A client base** — ``class WeatherProxy(
   A2AClient): ...``. If a URL literal appears in the class body we
   attach it; otherwise we emit the proxy with an ``unresolved_url``
   verification status.
4. **``/a2a`` / ``/a2a/v1`` path suffix literals** — only emit when the
   same file also contains a client-constructor call or a class
   inheritance signal. On their own they are too ambiguous to warrant
   a proxy component.

Once A2A candidates are collected we build a URL index from the current
scan's Agent Cards (via :func:`aibom.scanners.a2a_detector.
iter_scanned_agent_cards`) and resolve each candidate URL against it.
Unresolved candidates stay at ``unverified_url_pattern`` until the
post-scan cross-repo linker (Phase 4d) upgrades them to
``verified_cross_repo_card`` when a matching Agent Card is found in
another scanned repository.

Remote agent-runtime SDKs (fallback, only classes not already A2A-tagged)
-------------------------------------------------------------------------

For classes that the A2A signals do **not** catch, a second pass looks
for calls into SDKs where the *server* runs the agentic loop. The
catalog of recognized SDKs ships in
:mod:`aibom.agent_signatures` as :class:`RemoteAgentSdkSignature`
entries and can be extended by the user via ``agent_signatures.
remote_agent_sdks`` in ``.aibom.yaml``.

A class is promoted to ``AGENT_PROXY`` only when *all* of these hold:

* At least one ``call`` observation inside the class body has a
  :attr:`~aibom.structures.CallObservation.qualified_name` that matches
  (equals, or ends with ``.<suffix>``) one of the SDK's
  :attr:`~aibom.agent_signatures.RemoteAgentSdkSignature.
  call_qualified_suffixes`, **and**
* If the signature declares
  :attr:`~aibom.agent_signatures.RemoteAgentSdkSignature.
  import_substrings`, at least one import in the file matches one of
  them (so that short qualified names like ``ReasoningEngine`` don't
  match the wrong library).

This is deliberately narrower than "any HTTP/SSE call into an LLM
endpoint": plain ``openai.ChatCompletion.create`` / token-streaming
``chat.completions`` calls are NOT SDK matches, because the local
caller owns the control flow. We rely on explicit, documented
agent-runtime SDKs rather than URL path heuristics.

We skip classes that:

* already have an A2A-signal match anywhere in the same class body
  (first-preference rule),
* are decorated with a Temporal / Celery / Airflow / FastAPI endpoint
  anti-pattern decorator from the merged signature catalog,
* inherit from a framework-agent base class (already surfaced by the
  KB / config scanners), or
* match a pydantic-basemodel / workflow / DAG anti-pattern signature.

The classification decision still belongs to the LLM — this scanner
only surfaces a structurally plausible candidate so the Phase-5 prompt
can see it and the Phase-6 verification gate re-checks the citation.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

from pathspec import PathSpec

from ..agent_signatures import (
    AgentAntiPatternSignature,
    AgentSignatureCatalog,
    RemoteAgentSdkSignature,
    resolve_catalog,
)
from ..cst_parser import parse_source_code
from ..models import AIComponent, ComponentRelationship, ScanContext
from ..models.enums import (
    AIComponentType,
    DetectionSource,
    RelationshipType,
)
from ..structures import (
    CallObservation,
    ClassBodyFactsObservation,
    ClassDefObservation,
    CodeAnalysisResult,
)
from .a2a_detector import (
    _iter_files_under,
    _load_exclude_spec,
    iter_scanned_agent_cards,
)
from .base import BaseScanner
from .file_cache import is_python_source, read_python_source

_LOGGER = logging.getLogger(__name__)

_A2A_CLIENT_CONSTRUCTORS: frozenset[str] = frozenset({
    "A2AClient",
    "A2AAgentClient",
    "A2ACardResolver",
    "AgentClient",
    "A2ARemoteAgent",
})

_A2A_CLIENT_URL_KWARGS: tuple[str, ...] = (
    "url",
    "base_url",
    "endpoint",
    "agent_url",
    "server_url",
    "a2a_url",
    "agent_card_url",
)

_A2A_CLIENT_BASE_NAMES: frozenset[str] = frozenset({
    "A2AClient",
    "A2AAgentClient",
    "A2ACardResolver",
})

_PY_CLIENT_HINTS: tuple[str, ...] = (
    "A2AClient",
    "A2ACardResolver",
    "A2AAgentClient",
    "AgentClient",
    "A2ARemoteAgent",
    ".well-known/agent",
    "/a2a/",
    "/a2a\"",
    "/a2a'",
)

_WELL_KNOWN_URL_RE = re.compile(
    r"^https?://[^\s/]+(:\d+)?/\.well-known/agent(-card)?\.json$",
    re.IGNORECASE,
)

_A2A_PATH_URL_RE = re.compile(
    r"^https?://[^\s/]+(:\d+)?/a2a(/v\d+)?/?$",
    re.IGNORECASE,
)

_MAX_PY_FILE_SIZE_BYTES = 2 * 1024 * 1024


def _normalize_url_for_index(url: str) -> str:
    """Canonicalize ``url`` so proxy URLs match Agent Card endpoints.

    Folds host casing, strips default ports, removes trailing slashes
    (except the root path), drops query strings and fragments, and
    rewrites ``https://host/.well-known/agent.json`` /
    ``.../agent-card.json`` to the host's A2A service root
    (``https://host``) so that a proxy fetching the well-known card can
    match a server that advertises its card at the same host. All
    comparisons are done on the canonicalized form, never on raw input.
    """
    if not isinstance(url, str):
        return ""
    raw = url.strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"}:
        return ""
    host = (parsed.hostname or "").lower()
    if not host:
        return ""
    port = parsed.port
    if (parsed.scheme == "https" and port == 443) or (
        parsed.scheme == "http" and port == 80
    ):
        port = None
    netloc = host if port is None else f"{host}:{port}"
    path = parsed.path or ""
    lower = path.lower()
    if lower.endswith("/.well-known/agent.json"):
        path = path[: -len("/.well-known/agent.json")]
    elif lower.endswith("/.well-known/agent-card.json"):
        path = path[: -len("/.well-known/agent-card.json")]
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return urlunparse((parsed.scheme, netloc, path, "", "", ""))


def _url_is_well_known_agent_card(url: str) -> bool:
    return bool(_WELL_KNOWN_URL_RE.match(url.strip()))


def _url_has_a2a_path_suffix(url: str) -> bool:
    return bool(_A2A_PATH_URL_RE.match(url.strip()))


def _match_a2a_client_call(qn: str) -> bool:
    """Return True if ``qn`` is a recognized A2A client constructor tail."""
    if not qn:
        return False
    tail = qn.rsplit(".", 1)[-1]
    return tail in _A2A_CLIENT_CONSTRUCTORS


def _extract_url_from_call(
    call: CallObservation,
) -> Optional[str]:
    """Pull the most plausible URL literal out of a client constructor call.

    Prefers named kwargs (``url=``, ``base_url=``, etc.), then falls back
    to the first positional argument if it is a string literal. Values
    that the CST parser stored as ``VARIABLE:`` / ``ATTRIBUTE:`` / etc.
    placeholders are discarded — only fully resolved string literals
    count, because we cannot know their value without executing the
    code.
    """
    args = call.arguments or {}
    for kw in _A2A_CLIENT_URL_KWARGS:
        raw = args.get(kw)
        if isinstance(raw, str) and not _is_unresolved_symbol(raw):
            stripped = raw.strip()
            if stripped:
                return stripped
    raw = args.get("_pos_0")
    if isinstance(raw, str) and not _is_unresolved_symbol(raw):
        stripped = raw.strip()
        if stripped and "://" in stripped:
            return stripped
    return None


def _is_unresolved_symbol(value: str) -> bool:
    """CST sentinel prefixes for values whose literal is unknown."""
    return value.startswith(
        ("VARIABLE:", "ATTRIBUTE:", "CALL:", "UNHANDLED_NODE:")
    )


def _inherits_from_a2a_client(base_classes: list[str]) -> bool:
    """Detect ``class Foo(A2AClient):`` or ``class Foo(a2a.client.A2AClient):``."""
    for base in base_classes or []:
        if not base:
            continue
        tail = base.rsplit(".", 1)[-1]
        if tail in _A2A_CLIENT_BASE_NAMES:
            return True
    return False


def _proxy_name_from_url(url: str, fallback: str) -> str:
    """Synthesize a human-readable proxy name from ``url``'s host + path."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return fallback
    host = parsed.hostname or ""
    if not host:
        return fallback
    head_label = host.split(".")[0] or host
    head_label = head_label.replace("-", "_")
    return f"{head_label}_a2a_proxy"


def _py_text_preselect(text: str) -> bool:
    """Cheap substring filter: skip CST parsing for files with no A2A hints."""
    return any(hint in text for hint in _PY_CLIENT_HINTS)


def _iter_python_files(context: ScanContext) -> list[Path]:
    all_files: list[Path] = []
    idx = context.file_index()
    if idx:
        all_files = [e.path for entries in idx.values() for e in entries]
    else:
        spec = _load_exclude_spec(context.exclude_patterns)
        for raw in context.paths:
            root = Path(raw).expanduser()
            all_files.extend(_iter_files_under(root, spec))
    return [p for p in all_files if is_python_source(p)]


def _build_url_index(
    cards: list[AIComponent],
) -> dict[str, list[AIComponent]]:
    """Build a normalized-URL → Agent Card component map.

    Indexes every ``metadata['agent_card']['endpoints']`` URL under its
    normalized form, plus a synthetic ``http(s)://host`` entry (no path)
    so proxies pointing at the bare A2A root of a host match the card.
    Cards without any endpoint are skipped for lookup but are still
    reachable via :func:`iter_scanned_agent_cards`.
    """
    out: dict[str, list[AIComponent]] = {}
    for card in cards:
        endpoints = (
            card.metadata.get("agent_card", {}).get("endpoints") or []
        )
        for raw in endpoints:
            if not isinstance(raw, str):
                continue
            canon = _normalize_url_for_index(raw)
            if not canon:
                continue
            out.setdefault(canon, []).append(card)
            try:
                parsed = urlparse(canon)
            except ValueError:
                continue
            host_root = urlunparse(
                (parsed.scheme, parsed.netloc, "", "", "", "")
            )
            if host_root and host_root != canon:
                out.setdefault(host_root, []).append(card)
    deduped: dict[str, list[AIComponent]] = {}
    for key, comps in out.items():
        seen_ids: set[str] = set()
        unique: list[AIComponent] = []
        for c in comps:
            if c.instance_id in seen_ids:
                continue
            seen_ids.add(c.instance_id)
            unique.append(c)
        deduped[key] = unique
    return deduped


def _resolve_against_index(
    url: str, index: dict[str, list[AIComponent]]
) -> Optional[AIComponent]:
    """Return the first matching Agent Card component, if any.

    Matches the normalized proxy URL against the index. If no exact
    match and the URL ends in a canonical A2A path (``/a2a``,
    ``/a2a/vX``, or ``/.well-known/agent.json``), also tries the host
    root (``scheme://host``) which is what ``_normalize_url_for_index``
    registers for Agent Cards.
    """
    canon = _normalize_url_for_index(url)
    if not canon:
        return None
    matches = index.get(canon)
    if matches:
        return matches[0]
    try:
        parsed = urlparse(canon)
    except ValueError:
        return None
    host_root = urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))
    matches = index.get(host_root)
    if matches:
        return matches[0]
    return None


def _component_from_proxy(
    *,
    path: Path,
    line: int,
    remote_url: Optional[str],
    constructor: Optional[str],
    detection_reason: str,
    url_index: dict[str, list[AIComponent]],
    class_name: Optional[str] = None,
) -> AIComponent:
    resolved = (
        _resolve_against_index(remote_url, url_index) if remote_url else None
    )
    if resolved is not None:
        status = "verified_local_card"
        confidence = 1.0
        matched_id = resolved.instance_id
        matched_name = resolved.name
        match_source = "local_scan"
    elif remote_url:
        status = "unverified_url_pattern"
        confidence = 0.5
        matched_id = ""
        matched_name = ""
        match_source = ""
    else:
        status = "unresolved_url_missing"
        confidence = 0.3
        matched_id = ""
        matched_name = ""
        match_source = ""

    name_fallback = (
        class_name
        or (path.stem if path.stem else "a2a_proxy")
    )
    component_name = (
        _proxy_name_from_url(remote_url, name_fallback)
        if remote_url
        else f"{name_fallback}_a2a_proxy"
    )

    metadata: dict[str, Any] = {
        "remote_url": remote_url or "",
        "detection_reason": detection_reason,
        "remote_verification": {
            "status": status,
            "confidence": confidence,
            "matched_component_instance_id": matched_id,
            "matched_component_name": matched_name,
            "match_source": match_source,
        },
    }
    if constructor:
        metadata["constructor"] = constructor
    if class_name:
        metadata["class_name"] = class_name

    return AIComponent(
        name=component_name,
        component_type=AIComponentType.AGENT_PROXY,
        file_path=str(path.resolve()),
        line_number=line,
        framework="a2a",
        detection_source=DetectionSource.CODE_ANALYSIS,
        heuristic_confidence=confidence,
        needs_agentic=False,
        agentic_hint=(
            f"A2A remote-agent proxy site ({detection_reason}); "
            "does not itself implement an agent loop."
        ),
        metadata=metadata,
    )


def _candidates_from_calls(
    result: CodeAnalysisResult,
) -> list[tuple[CallObservation, Optional[str], str]]:
    """Collect proxy candidates from constructor calls (direct + assignments).

    Each tuple is ``(call, remote_url_or_none, constructor_name)``.
    """
    out: list[tuple[CallObservation, Optional[str], str]] = []
    seen_positions: set[tuple[int, str]] = set()

    def _consume(call: CallObservation) -> None:
        qn = call.qualified_name
        if not _match_a2a_client_call(qn):
            return
        key = (call.line_number, qn)
        if key in seen_positions:
            return
        seen_positions.add(key)
        ctor = qn.rsplit(".", 1)[-1] if qn else ""
        url = _extract_url_from_call(call)
        out.append((call, url, ctor))

    for call in result.calls:
        _consume(call)
    for assignment in result.assignments:
        _consume(assignment.call)
    return out


def _class_defs_inheriting_a2a_client(
    result: CodeAnalysisResult,
) -> list[tuple[str, int, Optional[str]]]:
    """Return ``(class_name, line, url_from_body_if_any)`` tuples."""
    out: list[tuple[str, int, Optional[str]]] = []
    for cls in result.class_defs or []:
        if not _inherits_from_a2a_client(cls.base_classes):
            continue
        url: Optional[str] = None
        facts = next(
            (
                f
                for f in (result.class_bodies or [])
                if f.class_name == cls.class_name
            ),
            None,
        )
        if facts and facts.body_source:
            for literal in _well_known_urls_in_text(facts.body_source):
                url = literal
                break
        out.append((cls.class_name, cls.line_number, url))
    return out


def _well_known_urls_in_text(text: str) -> list[str]:
    """Return every URL in ``text`` that matches a canonical A2A pattern."""
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for raw in _find_http_url_tokens(text):
        if raw in seen:
            continue
        seen.add(raw)
        if _url_is_well_known_agent_card(raw) or _url_has_a2a_path_suffix(raw):
            found.append(raw)
    return found


_URL_TOKEN_RE = re.compile(r"https?://[^\s'\"<>(),]+", re.IGNORECASE)


def _find_http_url_tokens(text: str) -> list[str]:
    return [m.group(0).rstrip(".,;") for m in _URL_TOKEN_RE.finditer(text)]


def _string_literal_well_known_candidates(
    result: CodeAnalysisResult,
) -> list[tuple[int, str]]:
    """Find ``.well-known/agent(-card).json`` URLs in string literals.

    Only canonical well-known Agent Card URLs; looser patterns like
    ``/a2a`` are handled by the constructor / base-class signals, where
    we have stronger context.
    """
    out: list[tuple[int, str]] = []
    for lit in result.protocol_strings or []:
        value = (lit.value or "").strip()
        if not value:
            continue
        if _url_is_well_known_agent_card(value):
            out.append((lit.line_number, value))
    return out


# ---------------------------------------------------------------------------
# Remote agent-runtime SDK fallback — only runs on classes A2A did not catch.
# ---------------------------------------------------------------------------
#
# This pass matches class bodies that call into an SDK where the *server*
# runs the agent loop (OpenAI Assistants, LangGraph Cloud, AWS Bedrock
# Agents Runtime, AWS Bedrock AgentCore Runtime, Vertex Reasoning
# Engines, and any user-extended SDKs from ``.aibom.yaml``'s
# ``agent_signatures.remote_agent_sdks`` list).
#
# Plain LLM-completion calls (``chat.completions.create``,
# ``openai.Completion.create``, direct ``invoke_model``) deliberately do
# NOT match: the local caller runs the control loop in those cases.
# Structural HTTP/SSE heuristics have been removed — they were too
# generic to distinguish "streaming LLM tokens" from "streaming a remote
# agent loop." SDK-name matching is narrower and more verifiable.


def _build_sdk_preselect_hints(catalog: AgentSignatureCatalog) -> tuple[str, ...]:
    """Build cheap substring hints used to skip files with no SDK presence.

    Pulls every ``import_substrings`` entry plus the last segment of each
    ``call_qualified_suffixes`` entry (e.g. ``.invoke_agent`` →
    ``invoke_agent``) from the catalog's SDK signatures. If none of the
    hints appear in the file text, the parsing pass is skipped.
    """
    hints: set[str] = set()
    for sdk in catalog.remote_agent_sdks or ():
        for sub in sdk.import_substrings or ():
            if sub:
                hints.add(sub)
        for suffix in sdk.call_qualified_suffixes or ():
            if not suffix:
                continue
            tail = suffix.rsplit(".", 1)[-1]
            if tail:
                hints.add(tail)
    return tuple(sorted(hints))


def _match_sdk(
    qualified_name: str, sdks: list[RemoteAgentSdkSignature]
) -> Optional[RemoteAgentSdkSignature]:
    """Return the first SDK whose call suffix matches *qualified_name*.

    A suffix ``"foo.bar"`` matches both ``qualified_name == "foo.bar"``
    and ``qualified_name.endswith(".foo.bar")``. A bare-name suffix like
    ``"RemoteGraph"`` also matches ``qualified_name == "RemoteGraph"`` or
    any qualified name whose last dotted segment equals it.
    """
    if not qualified_name:
        return None
    for sdk in sdks:
        for suffix in sdk.call_qualified_suffixes or ():
            if not suffix:
                continue
            if qualified_name == suffix:
                return sdk
            if suffix.startswith(".") and qualified_name.endswith(suffix):
                return sdk
            if "." not in suffix:
                tail = qualified_name.rsplit(".", 1)[-1]
                if tail == suffix:
                    return sdk
            elif qualified_name.endswith("." + suffix):
                return sdk
    return None


def _filter_sdks_by_imports(
    catalog: AgentSignatureCatalog, imports: list[tuple[int, str]]
) -> list[RemoteAgentSdkSignature]:
    """Return SDKs whose ``import_substrings`` appear in *imports*.

    Signatures with an empty ``import_substrings`` list always match
    (useful for user-defined internal SDKs that don't ship under a
    recognizable package name). All imports are compared case-insensitively.
    """
    imports_text = "\n".join(stmt or "" for _, stmt in (imports or [])).lower()
    active: list[RemoteAgentSdkSignature] = []
    for sdk in catalog.remote_agent_sdks or ():
        if not sdk.import_substrings:
            active.append(sdk)
            continue
        if any(sub.lower() in imports_text for sub in sdk.import_substrings):
            active.append(sdk)
    return active


def _class_has_anti_pattern_decorator(
    class_body_facts: ClassBodyFactsObservation,
    anti_patterns: list[AgentAntiPatternSignature],
) -> bool:
    """Return True if *class_body_facts* carries any anti-pattern decorator.

    We deliberately reuse the merged catalog's anti-patterns (Temporal,
    Airflow, FastAPI endpoint, pydantic base model, etc.) so that
    promoting a class to ``AGENT_PROXY`` never contradicts a verdict the
    evidence-builder would give the same class under the agent rubric.
    """
    decorators = class_body_facts.class_decorators or []
    for dec in decorators:
        for ap in anti_patterns or []:
            for wanted in ap.decorator_qualified_names or ():
                if dec == wanted or dec.endswith("." + wanted):
                    return True
    return False


def _class_inherits_anti_pattern_base(
    class_def: ClassDefObservation,
    anti_patterns: list[AgentAntiPatternSignature],
) -> bool:
    bases = class_def.base_classes or []
    for base in bases:
        if not base:
            continue
        for ap in anti_patterns or []:
            for wanted in ap.base_class_names or ():
                if base == wanted or base.endswith("." + wanted):
                    return True
    return False


def _emit_sdk_proxy(
    *,
    path: Path,
    class_def: ClassDefObservation,
    sdk: RemoteAgentSdkSignature,
    call: CallObservation,
) -> AIComponent:
    """Build an AGENT_PROXY component for a class that calls into *sdk*.

    Unlike the A2A code path, we attach the proxy to the class definition
    line (not the specific call site) because the class *as a whole* is
    the proxy: any of its methods can initiate the remote-agent request.
    Metadata captures both the SDK identifier and the concrete call-site
    anchor so the Phase-5 prompt and Phase-6 verification gate can
    re-check the citation.

    The remote endpoint URL is NOT recoverable from a static call alone
    (it comes from configuration), so the remote-verification status is
    always ``"unresolved_url_missing"`` here. The LLM stage receives the
    SDK identifier as the primary evidence.
    """
    sdk_slug = sdk.id.replace(".", "_")
    name = f"{class_def.class_name}_{sdk_slug}_proxy"
    metadata: dict[str, Any] = {
        "remote_url": "",
        "detection_reason": "remote_agent_sdk_call",
        "sdk_id": sdk.id,
        "sdk_name": sdk.sdk_name,
        "class_name": class_def.class_name,
        "call_qualified_name": call.qualified_name or "",
        "call_line": call.line_number,
        "remote_verification": {
            "status": "unresolved_url_missing",
            "confidence": 0.7,
            "matched_component_instance_id": "",
            "matched_component_name": "",
            "match_source": "",
        },
    }
    return AIComponent(
        name=name,
        component_type=AIComponentType.AGENT_PROXY,
        file_path=str(path.resolve()),
        line_number=class_def.line_number,
        framework=sdk.id,
        detection_source=DetectionSource.CODE_ANALYSIS,
        heuristic_confidence=0.7,
        needs_agentic=True,
        agentic_hint=(
            f"Invokes {sdk.sdk_name}; the remote service runs the "
            "agent loop. LLM should confirm this class is a proxy "
            "(not merely a plain completion client)."
        ),
        metadata=metadata,
    )


def _line_is_inside_any_class(
    line: int, class_bodies: list[ClassBodyFactsObservation]
) -> Optional[str]:
    """Return the class name whose body spans *line*, if any."""
    for cb in class_bodies or []:
        if cb.start_line <= line <= cb.end_line:
            return cb.class_name
    return None


def _scan_python_file_for_sdk_proxies(
    path: Path,
    result: CodeAnalysisResult,
    catalog: AgentSignatureCatalog,
    a2a_tagged_lines: frozenset[int],
) -> list[AIComponent]:
    """Return SDK-fallback AGENT_PROXY components for *result*.

    For each call observation whose qualified name matches one of the
    merged catalog's ``RemoteAgentSdkSignature`` entries, we emit an
    ``AGENT_PROXY`` attributed to the enclosing class (one proxy per
    ``(class, sdk)`` pair, first call wins).

    Classes that contain **any** A2A-tagged line anywhere in their body
    (class-def line, call-site inside ``__init__``, literal inside a
    method, etc.) are skipped so the A2A signal remains first-preference.
    Anti-patterns and framework inheritance are honored via *catalog*.
    Module-scope (non-class) calls are ignored because a standalone
    script that calls an SDK once doesn't define a "proxy class."
    """
    if not result.class_bodies or not catalog.remote_agent_sdks:
        return []

    active_sdks = _filter_sdks_by_imports(catalog, result.imports or [])
    if not active_sdks:
        return []

    class_def_map = {c.class_name: c for c in result.class_defs or []}
    class_bodies = list(result.class_bodies)
    body_facts_map = {f.class_name: f for f in class_bodies}

    a2a_tagged_classes: set[str] = set()
    for line in a2a_tagged_lines:
        cls_name = _line_is_inside_any_class(line, class_bodies)
        if cls_name:
            a2a_tagged_classes.add(cls_name)

    all_calls: list[CallObservation] = list(result.calls or [])
    for assignment in result.assignments or []:
        all_calls.append(assignment.call)

    out: list[AIComponent] = []
    seen: set[tuple[str, str]] = set()

    for call in all_calls:
        qn = call.qualified_name
        if not qn:
            continue
        sdk = _match_sdk(qn, active_sdks)
        if sdk is None:
            continue
        enclosing = _line_is_inside_any_class(call.line_number, class_bodies)
        if enclosing is None:
            continue
        if enclosing in a2a_tagged_classes:
            continue
        cls_def = class_def_map.get(enclosing)
        if cls_def is None:
            continue
        facts = body_facts_map.get(enclosing)
        if facts is not None and _class_has_anti_pattern_decorator(
            facts, catalog.anti_patterns
        ):
            continue
        if _class_inherits_anti_pattern_base(cls_def, catalog.anti_patterns):
            continue
        if _class_inherits_framework_agent(cls_def, catalog):
            continue
        if _inherits_from_a2a_client(cls_def.base_classes):
            continue

        dedup_key = (enclosing, sdk.id)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        out.append(_emit_sdk_proxy(path=path, class_def=cls_def, sdk=sdk, call=call))
    return out


def _class_inherits_framework_agent(
    class_def: ClassDefObservation,
    catalog: AgentSignatureCatalog,
) -> bool:
    """True if *class_def* inherits from any framework-agent base class."""
    bases = class_def.base_classes or []
    for base in bases:
        if not base:
            continue
        for fw in catalog.frameworks or []:
            for wanted in fw.base_class_names or ():
                if base == wanted or base.endswith("." + wanted):
                    return True
    return False


def _sdk_preselect(text: str, hints: tuple[str, ...]) -> bool:
    """Cheap gate: only run the SDK pass when at least one hint appears."""
    if not hints:
        return False
    return any(h in text for h in hints)


def _scan_python_file(
    path: Path,
    url_index: dict[str, list[AIComponent]],
    catalog: Optional[AgentSignatureCatalog] = None,
    sdk_preselect_hints: tuple[str, ...] = (),
) -> list[AIComponent]:
    try:
        if path.stat().st_size > _MAX_PY_FILE_SIZE_BYTES:
            return []
    except OSError:
        return []
    try:
        text = read_python_source(path)
    except OSError:
        return []

    want_a2a = _py_text_preselect(text)
    want_sdk = _sdk_preselect(text, sdk_preselect_hints)
    if not (want_a2a or want_sdk):
        return []

    try:
        result = parse_source_code(str(path), text)
    except Exception:
        _LOGGER.debug("CST parse failed for %s", path, exc_info=True)
        return []

    components: list[AIComponent] = []
    seen_dedup: set[tuple[int, str]] = set()
    a2a_class_lines: set[int] = set()

    def _emit(comp: AIComponent, dedup_url: str) -> None:
        key = (comp.line_number, dedup_url)
        if key in seen_dedup:
            return
        seen_dedup.add(key)
        components.append(comp)

    if want_a2a:
        for call, url, ctor in _candidates_from_calls(result):
            comp = _component_from_proxy(
                path=path,
                line=call.line_number,
                remote_url=url,
                constructor=ctor,
                detection_reason="client_constructor_call",
                url_index=url_index,
            )
            _emit(
                comp, _normalize_url_for_index(url or "") or f"line_{call.line_number}"
            )
            a2a_class_lines.add(call.line_number)

        for class_name, line, url in _class_defs_inheriting_a2a_client(result):
            comp = _component_from_proxy(
                path=path,
                line=line,
                remote_url=url,
                constructor=None,
                detection_reason="class_inherits_a2a_client_base",
                url_index=url_index,
                class_name=class_name,
            )
            _emit(comp, _normalize_url_for_index(url or "") or f"class_{class_name}")
            a2a_class_lines.add(line)

        for line, url in _string_literal_well_known_candidates(result):
            comp = _component_from_proxy(
                path=path,
                line=line,
                remote_url=url,
                constructor=None,
                detection_reason="well_known_url_literal",
                url_index=url_index,
            )
            _emit(comp, _normalize_url_for_index(url))
            a2a_class_lines.add(line)

    if want_sdk:
        resolved_catalog = catalog or resolve_catalog()
        sdk_candidates = _scan_python_file_for_sdk_proxies(
            path=path,
            result=result,
            catalog=resolved_catalog,
            a2a_tagged_lines=frozenset(a2a_class_lines),
        )
        for comp in sdk_candidates:
            sdk_id = comp.metadata.get("sdk_id") or ""
            class_name = comp.metadata.get("class_name") or ""
            dedup_key = f"sdk:{sdk_id}:{class_name}"
            _emit(comp, dedup_key)

    return components


def _relationships_for(
    proxies: list[AIComponent],
    url_index: dict[str, list[AIComponent]],
) -> list[ComponentRelationship]:
    """Produce ``INVOKES_A2A_AGENT`` edges for every verified proxy."""
    rels: list[ComponentRelationship] = []
    for proxy in proxies:
        verification = proxy.metadata.get("remote_verification") or {}
        if verification.get("status") != "verified_local_card":
            continue
        target_id = verification.get("matched_component_instance_id") or ""
        if not target_id:
            continue
        target_comp: Optional[AIComponent] = None
        for comps in url_index.values():
            for c in comps:
                if c.instance_id == target_id:
                    target_comp = c
                    break
            if target_comp:
                break
        if target_comp is None:
            continue
        rels.append(
            ComponentRelationship(
                source_instance_id=proxy.instance_id,
                target_instance_id=target_comp.instance_id,
                relationship_type=RelationshipType.INVOKES_A2A_AGENT,
                label="A2A proxy invokes local agent",
                source_name=proxy.name,
                target_name=target_comp.name,
                source_type=proxy.component_type,
                target_type=target_comp.component_type,
            )
        )
    return rels


class RemoteAgentResolver(BaseScanner):
    """Offline resolver for local code that invokes remote agents.

    Emits :data:`~aibom.models.enums.AIComponentType.AGENT_PROXY`
    components and :data:`~aibom.models.enums.RelationshipType.
    INVOKES_A2A_AGENT` relationships for every remote-agent invocation
    site. Two detection paths are supported:

    * **A2A signals (first preference).** Client-constructor calls,
      ``.well-known`` Agent Card URLs, A2A base-class inheritance, or
      ``/a2a`` suffix literals. Verified against Agent Cards in the
      current scan when possible; unresolved cases stay at
      ``unverified_url_pattern`` until the cross-repo linker (Phase 4d)
      matches them to a card elsewhere.
    * **Remote agent-runtime SDK fallback.** A class whose body calls
      into a known remote-agent SDK (OpenAI Assistants, LangGraph
      Cloud, AWS Bedrock Agents Runtime, AWS Bedrock AgentCore
      Runtime, Vertex Reasoning Engines, or user-defined SDKs via
      ``.aibom.yaml``) and is *not* already tagged by the A2A path.
      Guardrailed by the merged signature catalog's anti-patterns to
      avoid promoting Temporal workflows, pydantic models, FastAPI
      endpoints, framework agents, etc.
    """

    name = "remote_agent_resolver"

    def supports(self, context: ScanContext) -> bool:
        return True

    def scan(
        self, context: ScanContext
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        cards = iter_scanned_agent_cards(context)
        url_index = _build_url_index(cards)
        catalog = resolve_catalog()
        sdk_preselect_hints = _build_sdk_preselect_hints(catalog)

        proxies: list[AIComponent] = []
        for path in _iter_python_files(context):
            proxies.extend(
                _scan_python_file(
                    path,
                    url_index,
                    catalog=catalog,
                    sdk_preselect_hints=sdk_preselect_hints,
                )
            )

        final_seen: set[tuple[str, int, str]] = set()
        final: list[AIComponent] = []
        for comp in proxies:
            key = (
                comp.file_path,
                comp.line_number,
                comp.metadata.get("remote_url") or "",
            )
            if key in final_seen:
                continue
            final_seen.add(key)
            final.append(comp)

        rels = _relationships_for(final, url_index)
        _LOGGER.debug(
            "RemoteAgentResolver emitted %d proxies, %d relationships",
            len(final),
            len(rels),
        )
        _ = PathSpec
        return final, rels


__all__ = [
    "RemoteAgentResolver",
    "_normalize_url_for_index",
    "_url_is_well_known_agent_card",
    "_url_has_a2a_path_suffix",
    "_match_a2a_client_call",
    "_extract_url_from_call",
    "_inherits_from_a2a_client",
    "_build_url_index",
    "_resolve_against_index",
]
