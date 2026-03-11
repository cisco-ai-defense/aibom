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

"""Completeness scoring engine for AI BOM analysis results.

Validates that detected components have expected co-occurrences and
relationships, producing a score (0-100) and actionable warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from .structures import CategorizationOutput, ComponentRelationship


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

@dataclass
class ConceptRule:
    """Describes what a concept is expected to have."""
    expected_relationships: List[str] = field(default_factory=list)
    expected_co_concepts: List[str] = field(default_factory=list)
    warnings: Dict[str, str] = field(default_factory=dict)


COMPLETENESS_RULES: Dict[str, ConceptRule] = {
    "agent": ConceptRule(
        expected_relationships=["USES_LLM"],
        expected_co_concepts=["model"],
        warnings={
            "no_model": "Agent detected without an associated LLM/model -- likely missing a relationship",
            "no_prompt": "Agent detected without a visible prompt/template -- consider adding prompt detection",
        },
    ),
    "model": ConceptRule(
        expected_co_concepts=["agent"],
        warnings={
            "orphaned": "Model detected but not referenced by any agent or pipeline",
        },
    ),
    "retriever": ConceptRule(
        expected_relationships=["USES_EMBEDDING"],
        expected_co_concepts=["embedding", "datastore"],
        warnings={
            "no_embedding": "Retriever detected without an associated embedding model",
        },
    ),
    "datastore": ConceptRule(
        expected_relationships=["USES_EMBEDDING"],
        expected_co_concepts=["embedding"],
        warnings={
            "no_embedding": "Datastore detected without an associated embedding model",
        },
    ),
    "tool": ConceptRule(
        expected_co_concepts=["agent"],
        warnings={
            "orphaned": "Tool detected but not referenced by any agent",
        },
    ),
    "memory": ConceptRule(
        expected_co_concepts=["agent"],
        warnings={
            "orphaned": "Memory component detected but not referenced by any agent",
        },
    ),
    "prompt": ConceptRule(
        expected_co_concepts=["agent"],
        warnings={
            "orphaned": "Prompt template detected but not referenced by any agent",
        },
    ),
}


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class CompletenessReport:
    """Result of the completeness scoring."""
    score: float = 0.0
    warnings: List[str] = field(default_factory=list)
    breakdown: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 1),
            "warnings": self.warnings,
            "breakdown": self.breakdown,
        }


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

def compute_completeness_score(
    categorization_output: CategorizationOutput,
) -> CompletenessReport:
    """Evaluate how complete the detected AI BOM is.

    Scoring:
    - *Relationship coverage*: fraction of expected relationships satisfied.
    - *Co-concept coverage*: fraction of expected co-concepts present.
    - *Orphan penalty*: deduction for unconnected components.

    Returns a :class:`CompletenessReport` with a 0-100 score plus warnings.
    """
    components = categorization_output.components
    relationships = categorization_output.relationships

    present_concepts: Set[str] = set()
    for category in components:
        if components[category]:
            present_concepts.add(category.lower())

    rel_set = _relationship_index(relationships)

    total_checks = 0
    passed_checks = 0
    warnings: List[str] = []
    breakdown: Dict[str, Any] = {}

    for concept, rule in COMPLETENESS_RULES.items():
        concept_components = components.get(concept, [])
        if not concept_components:
            continue

        concept_stats = {"count": len(concept_components), "relationship_pass": 0, "co_concept_pass": 0, "total": 0}

        for comp in concept_components:
            instance_id = comp.get("instance_id", "")

            for rel_label in rule.expected_relationships:
                total_checks += 1
                concept_stats["total"] += 1
                if _has_relationship(instance_id, rel_label, rel_set):
                    passed_checks += 1
                    concept_stats["relationship_pass"] += 1

            for co_concept in rule.expected_co_concepts:
                total_checks += 1
                concept_stats["total"] += 1
                if co_concept.lower() in present_concepts:
                    passed_checks += 1
                    concept_stats["co_concept_pass"] += 1

        if concept == "agent":
            _check_agent_warnings(concept_components, rel_set, present_concepts, rule, warnings)
        elif rule.warnings:
            _check_generic_warnings(concept, concept_components, rel_set, present_concepts, rule, warnings)

        breakdown[concept] = concept_stats

    if total_checks == 0:
        score = 0.0 if not present_concepts else 100.0
    else:
        score = (passed_checks / total_checks) * 100.0

    return CompletenessReport(score=score, warnings=warnings, breakdown=breakdown)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _relationship_index(
    relationships: List[ComponentRelationship],
) -> Dict[str, Set[str]]:
    """Map source_instance_id -> set of relationship labels."""
    idx: Dict[str, Set[str]] = {}
    for rel in relationships:
        idx.setdefault(rel.source_instance_id, set()).add(rel.label)
        idx.setdefault(rel.target_instance_id, set()).add(f"_{rel.label}")
    return idx


def _has_relationship(instance_id: str, label: str, rel_set: Dict[str, Set[str]]) -> bool:
    labels = rel_set.get(instance_id, set())
    return label in labels


def _check_agent_warnings(
    components: List[Dict[str, Any]],
    rel_set: Dict[str, Set[str]],
    present_concepts: Set[str],
    rule: ConceptRule,
    warnings: List[str],
) -> None:
    for comp in components:
        iid = comp.get("instance_id", "")
        name = comp.get("name", iid)
        if not _has_relationship(iid, "USES_LLM", rel_set) and "model" not in present_concepts:
            warnings.append(f"[{name}] {rule.warnings.get('no_model', 'Missing model')}")
        if not _has_relationship(iid, "USES_PROMPT", rel_set) and "prompt" not in present_concepts:
            warnings.append(f"[{name}] {rule.warnings.get('no_prompt', 'Missing prompt')}")


def _check_generic_warnings(
    concept: str,
    components: List[Dict[str, Any]],
    rel_set: Dict[str, Set[str]],
    present_concepts: Set[str],
    rule: ConceptRule,
    warnings: List[str],
) -> None:
    for comp in components:
        iid = comp.get("instance_id", "")
        name = comp.get("name", iid)
        labels = rel_set.get(iid, set())
        is_referenced = any(lbl.startswith("_") for lbl in labels)
        if not is_referenced:
            for co in rule.expected_co_concepts:
                if co.lower() not in present_concepts:
                    msg = rule.warnings.get("orphaned", f"{concept} component not connected")
                    warnings.append(f"[{name}] {msg}")
                    break
