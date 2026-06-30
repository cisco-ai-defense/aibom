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

"""Optional ATR security-enrichment source.

Runs the open-source `Agent Threat Rules <https://pypi.org/project/pyatr/>`_
(``pyatr``) engine over the content of already-detected skill / prompt / agent /
MCP components and tags any component whose content matches a known
agent-attack rule with that rule's MITRE ATLAS (and ATT&CK, where the rule
carries one) technique IDs.

This is a *finding about one asset* -- "this SKILL.md looks like a weaponized
instruction -> AML.T0010" -- not a pass/fail posture check, so it lives here as
a per-asset enrichment rather than in the compliance layer.  The enrichment is
opt-in: a normal inventory run is unchanged, and the pass is a no-op when the
optional ``pyatr`` dependency is not installed, mirroring how
``skill_detector`` guards its optional skill-scanner call.

The technique IDs are written to ``component.metadata["security_enrichment"]``
so downstream reporting / crosswalk views can roll them up alongside the
compliance section without any of the detectors having to know about ATR.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from .models import AIComponent
from .models.enums import AIComponentType

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from pyatr import ATRMatch

_LOGGER = logging.getLogger(__name__)

# Config key consumed from ``ScanContext.config`` / the CLI flag.  When absent
# or falsey the enrichment pass does nothing, so default inventory runs are
# untouched.
CONFIG_KEY = "atr_enrichment"

# Optional override pointing at an ATR rules/ directory.  The per-rule MITRE
# ATLAS / ATT&CK ``references`` block is only present in the source YAML rules,
# not in the slim bundle that ``pyatr.load_default_rules()`` ships (the bundle
# keeps only the fields the engine evaluates).  Point this at an ATR checkout
# to surface technique IDs; without it, matches are still reported by rule id /
# title but carry empty technique lists.
RULES_DIR_ENV = "ATR_RULES_DIR"

# The asset surface ATR rules are written against: agent instructions, prompt
# text, MCP tool/server descriptions.  Other component types (models, vector
# stores, datasets, ...) are not in scope and are returned unchanged.
ENRICHABLE_TYPES: frozenset[AIComponentType] = frozenset(
    {
        AIComponentType.SKILL,
        AIComponentType.PROMPT,
        AIComponentType.AGENT,
        AIComponentType.MCP_SERVER,
        AIComponentType.MCP_CLIENT,
    }
)

# Reference keys on an ATR rule that carry technique IDs we surface.  ATLAS is
# the primary mapping for the AI-agent surface; ATT&CK is carried only on rules
# that have a classic-technique analogue, so it is emitted opportunistically.
_ATLAS_REF_KEYS: tuple[str, ...] = ("mitre_atlas", "atlas")
_ATTACK_REF_KEYS: tuple[str, ...] = ("mitre_attack", "attack", "mitre_att&ck")

# ATLAS technique ids look like ``AML.T0010``; ATT&CK like ``T1059`` or
# ``T1059.006``.  References are stored as ``"<ID> - <Name>"`` strings, so we
# pull the leading identifier rather than the human label.
_ATLAS_ID_RE = re.compile(r"\bAML\.T\d{4}(?:\.\d{3})?\b")
_ATTACK_ID_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")

# Cap on bytes read from a file-backed component (e.g. a SKILL.md) so a hostile
# or oversized asset can't blow up memory during enrichment.
_MAX_CONTENT_BYTES = 256 * 1024


def _load_pyatr() -> Any:
    """Return the ``pyatr`` module, or ``None`` when it is not installed.

    Kept as a thin indirection so the enrichment pass degrades to a no-op
    instead of raising when the optional dependency is absent.
    """
    try:
        import pyatr
    except ImportError:
        _LOGGER.debug("pyatr not installed; ATR security enrichment skipped")
        return None
    return pyatr


def _ids_from_refs(
    refs: Any, keys: tuple[str, ...], pattern: re.Pattern[str]
) -> list[str]:
    """Extract technique ids from an ATR rule ``references`` mapping.

    ``references`` is ``{key: ["<ID> - <Name>", ...]}``; values may also be a
    bare string.  Unknown shapes are ignored rather than raising.
    """
    if not isinstance(refs, dict):
        return []
    found: list[str] = []
    for key in keys:
        raw = refs.get(key)
        if raw is None:
            continue
        entries = raw if isinstance(raw, (list, tuple)) else [raw]
        for entry in entries:
            for match in pattern.findall(str(entry)):
                if match not in found:
                    found.append(match)
    return found


def _build_rule_reference_index(engine: Any) -> dict[str, dict[str, list[str]]]:
    """Map ``rule_id -> {"atlas": [...], "attack": [...]}`` from the engine.

    ``ATRMatch`` does not carry a rule's ``references`` block, so we read the
    loaded ``ATRRule`` objects (which do) once up front and index by id.
    """
    index: dict[str, dict[str, list[str]]] = {}
    for rule in getattr(engine, "rules", []) or []:
        refs = getattr(rule, "references", None)
        atlas = _ids_from_refs(refs, _ATLAS_REF_KEYS, _ATLAS_ID_RE)
        attack = _ids_from_refs(refs, _ATTACK_REF_KEYS, _ATTACK_ID_RE)
        if atlas or attack:
            index[rule.id] = {"atlas": atlas, "attack": attack}
    return index


def _read_file_content(file_path: str) -> str:
    """Best-effort read of a file-backed component's content (capped)."""
    if not file_path:
        return ""
    try:
        path = Path(file_path)
        if not path.is_file():
            return ""
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return fh.read(_MAX_CONTENT_BYTES)
    except OSError:
        return ""


def _component_content(component: AIComponent) -> str:
    """Assemble the text an ATR rule should be evaluated against.

    Pulls the in-memory content the detector already extracted (name,
    description, prompt text, skill trigger patterns) and, for file-backed
    assets such as a SKILL.md, the file body itself.
    """
    parts: list[str] = [component.name]
    if component.description:
        parts.append(component.description)
    if component.text:
        parts.append(component.text)

    triggers = component.metadata.get("trigger_patterns")
    if isinstance(triggers, (list, tuple)):
        parts.extend(str(t) for t in triggers)

    # Skills are config-file backed; read the SKILL.md / AGENTS.md body so the
    # rules see the actual instruction text, not just the parsed summary.
    if component.component_type == AIComponentType.SKILL and component.file_path:
        body = _read_file_content(component.file_path)
        if body:
            parts.append(body)

    return "\n".join(p for p in parts if p)


def _enrichment_for_content(
    content: str,
    engine: Any,
    ref_index: dict[str, dict[str, list[str]]],
) -> Optional[dict[str, Any]]:
    """Run the ATR engine over *content* and roll matches into a finding dict.

    Returns ``None`` when nothing matched, so callers can skip writing empty
    metadata onto clean components.
    """
    from pyatr import AgentEvent

    event = AgentEvent(
        content=content,
        event_type="llm_input",
        fields={"user_input": content},
    )
    matches: list[ATRMatch] = engine.evaluate(event)
    if not matches:
        return None

    atlas: list[str] = []
    attack: list[str] = []
    findings: list[dict[str, Any]] = []
    for match in matches:
        refs = ref_index.get(match.rule_id, {"atlas": [], "attack": []})
        m_atlas = refs.get("atlas", [])
        m_attack = refs.get("attack", [])
        for tid in m_atlas:
            if tid not in atlas:
                atlas.append(tid)
        for tid in m_attack:
            if tid not in attack:
                attack.append(tid)
        findings.append(
            {
                "rule_id": match.rule_id,
                "title": match.title,
                "severity": match.severity,
                "atlas_techniques": m_atlas,
                "attack_techniques": m_attack,
            }
        )

    return {
        "source": "atr",
        "atlas_techniques": atlas,
        "attack_techniques": attack,
        "findings": findings,
    }


def enrich_components(
    components: list[AIComponent],
    *,
    enabled: bool,
    rules_dir: str | Path | None = None,
) -> list[AIComponent]:
    """Tag matching skill / prompt / agent / MCP components with ATLAS IDs.

    Opt-in via *enabled*.  Returns the components unchanged when disabled, when
    ``pyatr`` is not installed, or when no in-scope component matches a rule.
    Matched components are returned as copies carrying a
    ``metadata["security_enrichment"]`` finding (immutable update; the inputs
    are not mutated).
    """
    if not enabled:
        return components

    pyatr = _load_pyatr()
    if pyatr is None:
        return components

    candidates = [c for c in components if c.component_type in ENRICHABLE_TYPES]
    if not candidates:
        return components

    resolved_rules_dir = rules_dir
    if resolved_rules_dir is None:
        env_dir = os.environ.get(RULES_DIR_ENV)
        if env_dir and Path(env_dir).is_dir():
            resolved_rules_dir = env_dir

    try:
        engine = pyatr.ATREngine()
        if resolved_rules_dir is not None:
            engine.load_rules_from_directory(Path(resolved_rules_dir))
        else:
            # The bundled rule set evaluates the same patterns but drops the
            # ATLAS/ATT&CK references, so matches are reported without technique
            # IDs unless ``ATR_RULES_DIR`` points at a rules checkout.
            engine.load_default_rules()
    except Exception:  # noqa: BLE001 - never let enrichment break a scan
        _LOGGER.debug("pyatr engine failed to load; enrichment skipped", exc_info=True)
        return components

    ref_index = _build_rule_reference_index(engine)

    enriched_by_id: dict[int, AIComponent] = {}
    match_count = 0
    for component in candidates:
        content = _component_content(component)
        if not content.strip():
            continue
        try:
            finding = _enrichment_for_content(content, engine, ref_index)
        except Exception:  # noqa: BLE001 - a bad rule must not break the scan
            _LOGGER.debug("ATR enrichment failed for %s", component.name, exc_info=True)
            continue
        if finding is None:
            continue
        new_meta = dict(component.metadata)
        new_meta["security_enrichment"] = finding
        enriched_by_id[id(component)] = component.model_copy(
            update={"metadata": new_meta}
        )
        match_count += 1

    if not enriched_by_id:
        return components

    _LOGGER.info(
        "ATR enrichment: tagged %d of %d in-scope component(s)",
        match_count,
        len(candidates),
    )
    return [enriched_by_id.get(id(c), c) for c in components]
