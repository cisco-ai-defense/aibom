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

"""Load and validate user-defined custom catalog configuration files.

Supports ``.aibom.yaml``, ``.aibom.yml``, and ``.aibom.json`` files that let
users register custom AI component symbols, base-class detection rules,
relationship hints, custom relationship types, and exclude patterns.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .agent_signatures import AgentSignatureCatalog, parse_user_signatures

LOGGER = logging.getLogger(__name__)

_CONFIG_FILENAMES = (".aibom.yaml", ".aibom.yml", ".aibom.json")


@dataclass
class CustomComponentEntry:
    """A single user-defined component catalog entry."""

    id: str
    concept: str
    label: str = ""
    framework: str = "custom"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_catalog_dict(self) -> Dict[str, Any]:
        """Convert to the dict format used by the supplemental catalog."""
        return {
            "id": self.id,
            "label": self.label or self.id.rsplit(".", 1)[-1],
            "concept": self.concept,
            "framework": self.framework,
            "sig_name": None,
            "type": None,
            "catalog_label": None,
            "_metadata": self.metadata,
        }


@dataclass
class BaseClassRule:
    """A rule that auto-categorizes any class inheriting from a given base."""

    base_class: str
    concept: str


@dataclass
class CustomRelationshipDef:
    """A user-defined relationship type with source/target constraints."""

    label: str
    source_categories: List[str] = field(default_factory=list)
    target_categories: List[str] = field(default_factory=list)
    argument_hints: List[str] = field(default_factory=list)


@dataclass
class CustomCatalogConfig:
    """The complete parsed result of a user custom catalog file."""

    components: List[CustomComponentEntry] = field(default_factory=list)
    base_class_rules: List[BaseClassRule] = field(default_factory=list)
    excludes: List[str] = field(default_factory=list)
    relationship_hints: Dict[str, List[str]] = field(default_factory=dict)
    custom_relationships: List[CustomRelationshipDef] = field(default_factory=list)
    agent_signatures: AgentSignatureCatalog = field(default_factory=AgentSignatureCatalog)
    source_path: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return (
            not self.components
            and not self.base_class_rules
            and not self.excludes
            and not self.relationship_hints
            and not self.custom_relationships
            and self.agent_signatures.is_empty
        )


_KNOWN_HINT_KEYS: Set[str] = {
    "tool_arguments",
    "llm_arguments",
    "memory_arguments",
    "retriever_arguments",
    "embedding_arguments",
}


def _parse_yaml_content(text: str) -> Any:
    """Parse YAML text using yaml.safe_load (safe deserialization only)."""
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required to load .aibom.yaml files. "
            "Install it with: pip install pyyaml"
        )
    return yaml.safe_load(text)


def _validate_component(raw: Dict[str, Any], index: int) -> Optional[CustomComponentEntry]:
    """Validate and convert a raw component dict from the config file."""
    comp_id = raw.get("id")
    concept = raw.get("concept")

    if not comp_id or not isinstance(comp_id, str):
        LOGGER.warning(
            "Skipping component at index %d: 'id' is required and must be a string.",
            index,
        )
        return None

    if not concept or not isinstance(concept, str):
        LOGGER.warning(
            "Skipping component '%s' at index %d: 'concept' is required and must be a string.",
            comp_id,
            index,
        )
        return None

    label = raw.get("label", "")
    if not isinstance(label, str):
        label = str(label)

    framework = raw.get("framework", "custom")
    if not isinstance(framework, str):
        framework = str(framework)

    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        LOGGER.warning(
            "Component '%s': 'metadata' must be a dict, ignoring provided value.",
            comp_id,
        )
        metadata = {}

    return CustomComponentEntry(
        id=comp_id,
        concept=concept.lower().strip(),
        label=label,
        framework=framework,
        metadata=metadata,
    )


def _validate_base_class_rule(raw: Dict[str, Any], index: int) -> Optional[BaseClassRule]:
    """Validate and convert a raw base_class rule dict."""
    base_class = raw.get("class")
    concept = raw.get("concept")

    if not base_class or not isinstance(base_class, str):
        LOGGER.warning(
            "Skipping base_class rule at index %d: 'class' is required.",
            index,
        )
        return None

    if not concept or not isinstance(concept, str):
        LOGGER.warning(
            "Skipping base_class rule for '%s': 'concept' is required.",
            base_class,
        )
        return None

    return BaseClassRule(base_class=base_class, concept=concept.lower().strip())


def _validate_custom_relationship(
    raw: Dict[str, Any], index: int
) -> Optional[CustomRelationshipDef]:
    """Validate a custom relationship definition."""
    label = raw.get("label")
    if not label or not isinstance(label, str):
        LOGGER.warning(
            "Skipping custom_relationship at index %d: 'label' is required.",
            index,
        )
        return None

    source_cats = raw.get("source_categories", [])
    target_cats = raw.get("target_categories", [])
    arg_hints = raw.get("argument_hints", [])

    if not isinstance(source_cats, list):
        source_cats = [source_cats] if source_cats else []
    if not isinstance(target_cats, list):
        target_cats = [target_cats] if target_cats else []
    if not isinstance(arg_hints, list):
        arg_hints = [arg_hints] if arg_hints else []

    return CustomRelationshipDef(
        label=label.upper(),
        source_categories=[str(c).lower().strip() for c in source_cats],
        target_categories=[str(c).lower().strip() for c in target_cats],
        argument_hints=[str(h).lower().strip() for h in arg_hints],
    )


def load_custom_catalog(config_path: Path) -> CustomCatalogConfig:
    """Load and validate a custom catalog file (YAML or JSON).

    Parameters
    ----------
    config_path:
        Explicit path to a ``.aibom.yaml``, ``.aibom.yml``, or ``.aibom.json`` file.

    Returns
    -------
    CustomCatalogConfig
        Parsed and validated configuration.  Returns an empty config on error.
    """
    if not config_path.is_file():
        LOGGER.warning("Custom catalog file not found: %s", config_path)
        return CustomCatalogConfig()

    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        LOGGER.warning("Could not read custom catalog %s: %s", config_path, exc)
        return CustomCatalogConfig()

    suffix = config_path.suffix.lower()
    try:
        if suffix in (".yaml", ".yml"):
            data = _parse_yaml_content(text)
        elif suffix == ".json":
            data = json.loads(text)
        else:
            LOGGER.warning(
                "Unsupported custom catalog format '%s'. Use .yaml, .yml, or .json.",
                suffix,
            )
            return CustomCatalogConfig()
    except Exception as exc:
        LOGGER.warning("Failed to parse custom catalog %s: %s", config_path, exc)
        return CustomCatalogConfig()

    if not isinstance(data, dict):
        LOGGER.warning("Custom catalog root must be a mapping, got %s.", type(data).__name__)
        return CustomCatalogConfig()

    return _build_config(data, source_path=str(config_path))


def discover_custom_catalog(project_root: Path) -> Optional[Path]:
    """Search *project_root* for a custom catalog file using the well-known names.

    Returns the first match or ``None``.
    """
    for name in _CONFIG_FILENAMES:
        candidate = project_root / name
        if candidate.is_file():
            LOGGER.info("Discovered custom catalog: %s", candidate)
            return candidate
    return None


def _build_config(data: Dict[str, Any], source_path: str = "") -> CustomCatalogConfig:
    """Build a ``CustomCatalogConfig`` from a parsed dict."""
    config = CustomCatalogConfig(source_path=source_path)

    # -- components --
    raw_components = data.get("components", [])
    if isinstance(raw_components, list):
        for idx, raw in enumerate(raw_components):
            if isinstance(raw, dict):
                entry = _validate_component(raw, idx)
                if entry:
                    config.components.append(entry)

    # -- base_classes --
    raw_base_classes = data.get("base_classes", [])
    if isinstance(raw_base_classes, list):
        for idx, raw in enumerate(raw_base_classes):
            if isinstance(raw, dict):
                rule = _validate_base_class_rule(raw, idx)
                if rule:
                    config.base_class_rules.append(rule)

    # -- excludes --
    raw_excludes = data.get("excludes", [])
    if isinstance(raw_excludes, list):
        config.excludes = [str(e) for e in raw_excludes if e]

    # -- relationship_hints --
    raw_hints = data.get("relationship_hints", {})
    if isinstance(raw_hints, dict):
        for key, values in raw_hints.items():
            if key not in _KNOWN_HINT_KEYS:
                LOGGER.warning(
                    "Unknown relationship_hints key '%s'. "
                    "Known keys: %s",
                    key,
                    ", ".join(sorted(_KNOWN_HINT_KEYS)),
                )
                continue
            if isinstance(values, list):
                config.relationship_hints[key] = [str(v).lower().strip() for v in values]

    # -- custom_relationships --
    raw_custom_rels = data.get("custom_relationships", [])
    if isinstance(raw_custom_rels, list):
        for idx, raw in enumerate(raw_custom_rels):
            if isinstance(raw, dict):
                rel_def = _validate_custom_relationship(raw, idx)
                if rel_def:
                    config.custom_relationships.append(rel_def)

    # -- agent_signatures --
    # Parsed only if the user supplies the section. The built-in catalog
    # of framework/protocol/anti-pattern signatures lives in
    # aibom.agent_signatures and is merged on top of this user catalog by
    # the evidence builder, not here.
    if "agent_signatures" in data:
        config.agent_signatures = parse_user_signatures(data.get("agent_signatures"))

    n_items = (
        len(config.components)
        + len(config.base_class_rules)
        + len(config.excludes)
        + len(config.custom_relationships)
        + sum(len(v) for v in config.relationship_hints.values())
    )
    sigs = config.agent_signatures
    n_sigs = (
        len(sigs.frameworks)
        + len(sigs.protocols)
        + len(sigs.anti_patterns)
        + len(sigs.remote_agent_sdks)
    )
    if n_items or n_sigs:
        LOGGER.info(
            "Loaded custom catalog from %s: %d component(s), %d base-class rule(s), "
            "%d exclude(s), %d custom relationship(s), "
            "%d framework sig(s), %d protocol sig(s), %d anti-pattern(s), "
            "%d remote agent SDK sig(s).",
            source_path,
            len(config.components),
            len(config.base_class_rules),
            len(config.excludes),
            len(config.custom_relationships),
            len(sigs.frameworks),
            len(sigs.protocols),
            len(sigs.anti_patterns),
            len(sigs.remote_agent_sdks),
        )

    return config


def parse_inline_annotation(comment_text: str) -> Optional[Dict[str, str]]:
    """Parse an ``# aibom: concept=model [framework=X] [label=Y]`` comment.

    Returns a dict of parsed key=value pairs, or ``None`` if the comment does
    not contain an aibom annotation.
    """
    if not comment_text:
        return None

    match = re.search(r"#\s*aibom:\s*(.+)", comment_text)
    if not match:
        return None

    payload = match.group(1).strip()
    result: Dict[str, str] = {}
    for token in re.findall(r"(\w+)=(\S+)", payload):
        result[token[0].lower()] = token[1].strip()

    if "concept" not in result:
        LOGGER.debug("aibom annotation missing 'concept': %s", comment_text)
        return None

    result["concept"] = result["concept"].lower().strip()
    return result
