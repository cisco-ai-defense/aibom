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

"""Entity-level evaluation for deterministic-to-agentic AIBOM decisions.

The existing :mod:`aibom.benchmark` module intentionally remains a lightweight,
report-level count/name benchmark.  This module evaluates the finer-grained
decision boundary used by the agentic pipeline:

* exact component, relationship, and risk identity precision/recall/F1;
* component and relationship recall lift over deterministic baselines;
* genuinely new-component discovery precision/recall;
* over-pruning of valid deterministic candidates; and
* keep/remove/enrich/reclassify action accuracy, macro-F1, and coverage.

Inputs may be AIBOM Pydantic models or their serialized dictionaries.  Outputs
are Pydantic models with deterministically sorted identifier lists, plus a flat
numeric projection suitable for local/custom evaluation integrations.
"""

from __future__ import annotations

import json
import posixpath
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

_CASE_ID_FIELDS = ("stable_case_id", "case_id", "eval_case_id")
_ACTION_VALUES = frozenset({"keep", "remove", "enrich", "reclassify", "discover"})
_DecisionAction = Literal["keep", "remove", "enrich", "reclassify", "discover"]

# Fields changed as a side effect of running the agentic pipeline, rather than
# fields that describe the component itself.  They must not turn an otherwise
# unchanged keep decision into an inferred enrichment.
_COMPONENT_BOOKKEEPING_FIELDS = frozenset(
    {
        "agent_evidence",
        "agentic_confidence",
        "agentic_hint",
        "decision_annotation",
        "id",
        "instance_id",
        "needs_agentic",
    }
)
_COMPONENT_IDENTITY_FIELDS = frozenset(
    {
        *_CASE_ID_FIELDS,
        "component_type",
        "file",
        "file_path",
        "line",
        "line_number",
        "name",
        "repo",
        "repository",
        "source_path",
        "source_repo",
        "type",
    }
)
_METADATA_BOOKKEEPING_FIELDS = frozenset(
    {
        *_CASE_ID_FIELDS,
        "agent_evidence",
        "agentic_confidence",
        "agentic_hint",
        "needs_agentic",
        "repo",
        "repository",
        "source_repo",
    }
)

# Canonical component identity is intentionally narrow.  These additional
# fields capture substantive enrichment while supplying model defaults so a
# sparse dictionary and an equivalent AIComponent model compare equally.
_COMPONENT_ENRICHMENT_DEFAULTS: dict[str, Any] = {
    "config_source": None,
    "dataset_source": None,
    "description": None,
    "detection_source": "code_analysis",
    "embedding_model": None,
    "framework": "",
    "heuristic_confidence": 1.0,
    "hyperparameters": {},
    "kb_concept": None,
    "kb_label": None,
    "metadata": {},
    "metrics": {},
    "model_name": None,
    "sdk_version": None,
    "skill_format": None,
    "storage_uri": None,
    "text": None,
    "training_info": None,
    "transport": None,
}


def _value(obj: Any, *names: str, default: Any = None) -> Any:
    """Return the first present mapping key or object attribute in *names*."""
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _enum_value(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value
    return "" if value is None else str(value)


def _normalise_token(value: Any) -> str:
    """Case-insensitive canonical form for semantic names and enum values."""
    return " ".join(_enum_value(value).strip().split()).casefold()


def _normalise_label_token(value: Any) -> str:
    """Canonical enum/flag label across case and common word separators."""
    words = _normalise_token(value).replace("-", " ").replace("_", " ").split()
    return "_".join(words)


def _metadata(component: Any) -> Mapping[str, Any]:
    raw = _value(component, "metadata", default={})
    return raw if isinstance(raw, Mapping) else {}


def _stable_case_id(component: Any) -> str:
    """Read the explicit evaluation case id, preferring top-level fields."""
    for field_name in _CASE_ID_FIELDS:
        raw = _value(component, field_name)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    metadata = _metadata(component)
    for field_name in _CASE_ID_FIELDS:
        raw = metadata.get(field_name)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    return ""


def _identifier_values(obj: Any, *names: str) -> list[str]:
    """Return unique, non-empty identifiers from *names* in preference order."""
    values: list[str] = []
    for name in names:
        candidate = _enum_value(_value(obj, name)).strip()
        if candidate and candidate not in values:
            values.append(candidate)
    return values


def _normalise_repository(value: Any) -> str:
    raw = _enum_value(value).strip().replace("\\", "/").rstrip("/")
    if not raw:
        return ""
    # Absolute source-repo paths are machine-specific.  Their terminal repo
    # directory is the stable repository label; explicit logical labels pass
    # through unchanged.
    if raw.startswith("/") or (len(raw) > 2 and raw[1] == ":"):
        raw = raw.rsplit("/", 1)[-1]
    return raw.casefold()


def _component_repository(
    component: Any,
    *,
    repository: str | None,
    repo_root: str | Path | None,
) -> str:
    if repository:
        return _normalise_repository(repository)
    direct = _value(component, "repository", "repo", "source_repo")
    if direct:
        return _normalise_repository(direct)
    metadata = _metadata(component)
    for key in ("repository", "repo", "source_repo"):
        if metadata.get(key):
            return _normalise_repository(metadata[key])
    if repo_root is not None:
        return _normalise_repository(Path(repo_root).name)
    return ""


def _repo_relative_path(raw_path: Any, repo_root: str | Path | None) -> str:
    """Return a slash-normalized path, relative to *repo_root* when possible."""
    text = _enum_value(raw_path).strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    if repo_root is not None:
        root = Path(repo_root).expanduser()
        try:
            if path.is_absolute():
                path = path.resolve(strict=False).relative_to(
                    root.resolve(strict=False)
                )
            else:
                # Serialized reports normally already contain a repo-relative
                # path.  Do not prepend the root and make it machine-specific.
                path = Path(str(path).replace("\\", "/"))
        except (OSError, ValueError):
            # Keep an out-of-root path explicit rather than collapsing it to a
            # basename and risking identity collisions.
            pass
    normalized = posixpath.normpath(str(path).replace("\\", "/"))
    return "" if normalized == "." else normalized


@dataclass(frozen=True, order=True)
class ComponentIdentity:
    """Canonical, hashable component identity.

    ``case_id`` is included when supplied, while type/name/location remain part
    of the exact identity.  This lets an evaluation distinguish a corrected
    reclassification from an unchanged candidate while still aligning both via
    :func:`component_action_key`.
    """

    component_type: str
    name: str
    repository: str
    source_path: str
    line_number: int
    case_id: str = ""

    @property
    def key(self) -> str:
        return json.dumps(
            {
                "case_id": self.case_id,
                "line": self.line_number,
                "name": self.name,
                "path": self.source_path,
                "repository": self.repository,
                "type": self.component_type,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, order=True)
class RelationshipIdentity:
    """Canonical directed relationship identity."""

    relationship_type: str
    source: str
    target: str

    @property
    def key(self) -> str:
        return json.dumps(
            {
                "source": self.source,
                "target": self.target,
                "type": self.relationship_type,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, order=True)
class RiskIdentity:
    """Canonical risk identity, excluding mutable prose and score weights."""

    flag: str
    severity: str
    repository: str
    source_path: str
    line_number: int
    case_id: str = ""

    @property
    def key(self) -> str:
        return json.dumps(
            {
                "case_id": self.case_id,
                "flag": self.flag,
                "line": self.line_number,
                "path": self.source_path,
                "repository": self.repository,
                "severity": self.severity,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


def canonical_component_identity(
    component: Any,
    *,
    repo_root: str | Path | None = None,
    repository: str | None = None,
) -> ComponentIdentity:
    """Build the exact identity used for component set comparisons."""
    raw_line = _value(component, "line_number", "line", default=0)
    try:
        line_number = int(raw_line or 0)
    except (TypeError, ValueError):
        line_number = 0
    return ComponentIdentity(
        component_type=_normalise_token(
            _value(component, "component_type", "type", default="other")
        ),
        name=_normalise_token(_value(component, "name", default="")),
        repository=_component_repository(
            component, repository=repository, repo_root=repo_root
        ),
        source_path=_repo_relative_path(
            _value(component, "file_path", "file", "source_path", default=""),
            repo_root,
        ),
        line_number=line_number,
        case_id=_stable_case_id(component),
    )


def component_action_key(
    component: Any,
    *,
    repo_root: str | Path | None = None,
    repository: str | None = None,
) -> str:
    """Stable key for aligning one candidate before and after a decision.

    An explicit stable case id wins.  Otherwise type is deliberately excluded,
    allowing a deterministic candidate and its reclassified result to align by
    normalized name and repo-relative source location.
    """
    identity = canonical_component_identity(
        component, repo_root=repo_root, repository=repository
    )
    if identity.case_id:
        return f"case:{identity.case_id}"
    return "loc:" + json.dumps(
        {
            "line": identity.line_number,
            "name": identity.name,
            "path": identity.source_path,
            "repository": identity.repository,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _component_lookups(
    components: Iterable[Any],
    *,
    repo_root: str | Path | None,
    repository: str | None,
) -> tuple[
    dict[str, ComponentIdentity],
    dict[tuple[str, str], list[ComponentIdentity]],
    dict[str, list[ComponentIdentity]],
]:
    by_identifier: dict[str, ComponentIdentity] = {}
    by_name_type: dict[tuple[str, str], list[ComponentIdentity]] = {}
    by_name: dict[str, list[ComponentIdentity]] = {}
    for component in components:
        identity = canonical_component_identity(
            component, repo_root=repo_root, repository=repository
        )
        identifiers = _identifier_values(
            component, *_CASE_ID_FIELDS, "instance_id", "id"
        )
        stable_case_id = _stable_case_id(component)
        if stable_case_id and stable_case_id not in identifiers:
            identifiers.append(stable_case_id)
        for identifier in identifiers:
            previous = by_identifier.get(identifier)
            if previous is None or identity.key < previous.key:
                by_identifier[identifier] = identity
        name_type_candidates = by_name_type.setdefault(
            (identity.name, identity.component_type), []
        )
        if identity not in name_type_candidates:
            name_type_candidates.append(identity)
        name_candidates = by_name.setdefault(identity.name, [])
        if identity not in name_candidates:
            name_candidates.append(identity)
    return by_identifier, by_name_type, by_name


def _relationship_endpoint(
    relationship: Any,
    side: Literal["source", "target"],
    *,
    by_identifier: Mapping[str, ComponentIdentity],
    by_name_type: Mapping[tuple[str, str], list[ComponentIdentity]],
    by_name: Mapping[str, list[ComponentIdentity]],
    repo_root: str | Path | None,
    repository: str | None,
) -> str:
    def _resolved_key(identity: ComponentIdentity) -> str:
        # Relationship identity follows the stable endpoint identifiers, not
        # mutable component attributes. In particular, a tool -> agent
        # reclassification must not make an otherwise unchanged baseline edge
        # look absent and manufacture relationship recall lift.
        if identity.case_id:
            return json.dumps(
                {"case_id": identity.case_id},
                sort_keys=True,
                separators=(",", ":"),
            )
        return identity.key

    nested = _value(relationship, f"{side}_component")
    if nested is not None:
        nested_case_id = _stable_case_id(nested)
        if nested_case_id:
            return json.dumps(
                {"case_id": nested_case_id},
                sort_keys=True,
                separators=(",", ":"),
            )
        nested_ids = _identifier_values(nested, "instance_id", "id")
        if nested_ids:
            return json.dumps(
                {"id": nested_ids[0]}, sort_keys=True, separators=(",", ":")
            )
        return _resolved_key(
            canonical_component_identity(
                nested, repo_root=repo_root, repository=repository
            )
        )

    endpoint_ids = _identifier_values(
        relationship,
        f"{side}_case_id",
        f"{side}_instance_id",
        f"{side}_id",
    )
    for endpoint_id in endpoint_ids:
        resolved = by_identifier.get(endpoint_id)
        if resolved is not None:
            if resolved.case_id:
                return _resolved_key(resolved)
            return json.dumps(
                {"id": endpoint_id}, sort_keys=True, separators=(",", ":")
            )

    # An explicit but unknown identifier is still meaningful.  Preserve it so
    # unrelated unresolved endpoints cannot collapse into the same blank
    # name/type fallback identity and produce a false relationship match.
    if endpoint_ids:
        return json.dumps(
            {"id": endpoint_ids[0]}, sort_keys=True, separators=(",", ":")
        )

    name = _normalise_token(_value(relationship, f"{side}_name", side, default=""))
    component_type = _normalise_token(
        _value(relationship, f"{side}_type", default="other")
    )
    exact_candidates = by_name_type.get((name, component_type), [])
    if len(exact_candidates) == 1:
        return exact_candidates[0].key
    name_candidates = by_name.get(name, [])
    if len(name_candidates) == 1:
        return name_candidates[0].key

    side_repository = _normalise_repository(
        _value(relationship, f"{side}_repo", default=repository or "")
    )
    return json.dumps(
        {
            "name": name,
            "repository": side_repository,
            "type": component_type,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_relationship_identity(
    relationship: Any,
    *,
    components: Iterable[Any] = (),
    repo_root: str | Path | None = None,
    repository: str | None = None,
) -> RelationshipIdentity:
    """Build a directed relationship identity using component identities.

    Stable case and runtime endpoint ids are resolved against *components*.
    Explicit ids that cannot be resolved remain part of the endpoint identity;
    when no id is supplied, unique type/name or name-only lookup is attempted
    before falling back to a deterministic endpoint identity.
    """
    component_list = list(components)
    by_identifier, by_name_type, by_name = _component_lookups(
        component_list, repo_root=repo_root, repository=repository
    )
    relation_type = _normalise_token(
        _value(relationship, "relationship_type", "label", "type", default="custom")
    )
    source = _relationship_endpoint(
        relationship,
        "source",
        by_identifier=by_identifier,
        by_name_type=by_name_type,
        by_name=by_name,
        repo_root=repo_root,
        repository=repository,
    )
    target = _relationship_endpoint(
        relationship,
        "target",
        by_identifier=by_identifier,
        by_name_type=by_name_type,
        by_name=by_name,
        repo_root=repo_root,
        repository=repository,
    )
    return RelationshipIdentity(relation_type, source, target)


def canonical_risk_identity(
    risk: Any,
    *,
    repo_root: str | Path | None = None,
    repository: str | None = None,
) -> RiskIdentity:
    """Build the exact identity used for risk set comparisons.

    Risk prose and scoring weights are intentionally excluded.  A nested
    ``component``/``candidate`` may supply source location and stable case ID
    when those fields are not repeated on the risk label itself.
    """
    nested = _value(risk, "component", "candidate", "source_component")
    raw_path = _value(risk, "file_path", "file", "source_path")
    if not _enum_value(raw_path).strip() and nested is not None:
        raw_path = _value(nested, "file_path", "file", "source_path", default="")

    raw_line = _value(risk, "line_number", "line")
    if raw_line in (None, "", 0, "0") and nested is not None:
        raw_line = _value(nested, "line_number", "line", default=0)
    try:
        line_number = int(raw_line or 0)
    except (TypeError, ValueError):
        line_number = 0

    if repository:
        risk_repository = _normalise_repository(repository)
    else:
        risk_repository = _component_repository(risk, repository=None, repo_root=None)
        if not risk_repository and nested is not None:
            risk_repository = _component_repository(
                nested, repository=None, repo_root=None
            )
        if not risk_repository and repo_root is not None:
            risk_repository = _normalise_repository(Path(repo_root).name)

    case_id = _stable_case_id(risk)
    if not case_id and nested is not None:
        case_id = _stable_case_id(nested)

    return RiskIdentity(
        flag=_normalise_label_token(
            _value(risk, "flag", "risk_flag", "risk_type", "type", "name", default="")
        ),
        severity=_normalise_label_token(_value(risk, "severity", default="")),
        repository=risk_repository,
        source_path=_repo_relative_path(raw_path, repo_root),
        line_number=line_number,
        case_id=case_id,
    )


class PRF1Metric(BaseModel):
    """Exact unique-entity set metrics."""

    model_config = ConfigDict(frozen=True)

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    predicted_count: int
    expected_count: int


def _empty_prf1_metric() -> PRF1Metric:
    return PRF1Metric(
        true_positives=0,
        false_positives=0,
        false_negatives=0,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        predicted_count=0,
        expected_count=0,
    )


class SetComparison(BaseModel):
    """Deterministically sorted identities behind one PRF1 result."""

    model_config = ConfigDict(frozen=True)

    true_positive_ids: list[str] = Field(default_factory=list)
    false_positive_ids: list[str] = Field(default_factory=list)
    false_negative_ids: list[str] = Field(default_factory=list)


class OverPruneMetric(BaseModel):
    """Valid baseline candidates incorrectly absent from the final output."""

    model_config = ConfigDict(frozen=True)

    over_pruned_count: int
    eligible_baseline_count: int
    rate: float
    over_pruned_action_keys: list[str] = Field(default_factory=list)


class ActionDecision(BaseModel):
    """Expected or predicted action for one stable candidate key."""

    model_config = ConfigDict(frozen=True)

    action: _DecisionAction
    target_type: str = ""


class ActionMismatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    action_key: str
    expected: ActionDecision
    predicted: ActionDecision | None = None


class AccuracyMetric(BaseModel):
    model_config = ConfigDict(frozen=True)

    correct_count: int
    evaluated_count: int
    accuracy: float
    mismatches: list[ActionMismatch] = Field(default_factory=list)
    unexpected_action_keys: list[str] = Field(default_factory=list)


class DecisionEvaluationDetails(BaseModel):
    """Auditable exact entity sets used by the aggregate metrics."""

    model_config = ConfigDict(frozen=True)

    components: SetComparison
    relationships: SetComparison
    risks: SetComparison = Field(default_factory=SetComparison)
    baseline_components: SetComparison | None = None
    baseline_relationships: SetComparison | None = None
    discoveries: SetComparison | None = None


class DecisionEvaluationResult(BaseModel):
    """Complete deterministic output of :func:`evaluate_decisions`."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["aibom.decision_evaluation.v1"] = (
        "aibom.decision_evaluation.v1"
    )
    components: PRF1Metric
    relationships: PRF1Metric
    risks: PRF1Metric = Field(default_factory=_empty_prf1_metric)
    baseline_components: PRF1Metric | None = None
    baseline_relationships: PRF1Metric | None = None
    net_recall_lift: float | None = None
    relationship_recall_lift: float | None = None
    discoveries: PRF1Metric | None = None
    over_pruning: OverPruneMetric | None = None
    action_accuracy: AccuracyMetric | None = None
    action_macro_f1: float | None = None
    decision_coverage: float | None = None
    reclassification_accuracy: AccuracyMetric | None = None
    details: DecisionEvaluationDetails

    def to_galileo_metrics(self) -> dict[str, float]:
        """Return stable, flat numeric values for custom/local metric logging."""
        values: dict[str, float] = {}

        def add_prf(prefix: str, metric: PRF1Metric) -> None:
            values[f"{prefix}.precision"] = metric.precision
            values[f"{prefix}.recall"] = metric.recall
            values[f"{prefix}.f1"] = metric.f1
            values[f"{prefix}.true_positives"] = float(metric.true_positives)
            values[f"{prefix}.false_positives"] = float(metric.false_positives)
            values[f"{prefix}.false_negatives"] = float(metric.false_negatives)

        add_prf("aibom.components", self.components)
        add_prf("aibom.relationships", self.relationships)
        add_prf("aibom.risks", self.risks)
        if self.baseline_components is not None:
            add_prf("aibom.baseline_components", self.baseline_components)
        if self.baseline_relationships is not None:
            add_prf("aibom.baseline_relationships", self.baseline_relationships)
        if self.net_recall_lift is not None:
            values["aibom.net_recall_lift"] = self.net_recall_lift
        if self.relationship_recall_lift is not None:
            values["aibom.relationship_recall_lift"] = self.relationship_recall_lift
        if self.discoveries is not None:
            add_prf("aibom.discoveries", self.discoveries)
        if self.over_pruning is not None:
            values["aibom.over_prune_rate"] = self.over_pruning.rate
            values["aibom.over_pruned_count"] = float(
                self.over_pruning.over_pruned_count
            )
        if self.action_accuracy is not None:
            values["aibom.action_accuracy"] = self.action_accuracy.accuracy
        if self.action_macro_f1 is not None:
            values["aibom.action_macro_f1"] = self.action_macro_f1
        if self.decision_coverage is not None:
            values["aibom.decision_coverage"] = self.decision_coverage
        if self.reclassification_accuracy is not None:
            values["aibom.reclassification_accuracy"] = (
                self.reclassification_accuracy.accuracy
            )
        return values


def _compare_sets(
    predicted: set[str], expected: set[str]
) -> tuple[PRF1Metric, SetComparison]:
    true_positive = predicted & expected
    false_positive = predicted - expected
    false_negative = expected - predicted
    tp = len(true_positive)
    fp = len(false_positive)
    fn = len(false_negative)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (
        PRF1Metric(
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            predicted_count=len(predicted),
            expected_count=len(expected),
        ),
        SetComparison(
            true_positive_ids=sorted(true_positive),
            false_positive_ids=sorted(false_positive),
            false_negative_ids=sorted(false_negative),
        ),
    )


def _identity_and_anchor_maps(
    components: Iterable[Any],
    *,
    repo_root: str | Path | None,
    repository: str | None,
) -> tuple[
    dict[str, ComponentIdentity],
    dict[str, ComponentIdentity],
    dict[str, str],
]:
    by_identity: dict[str, ComponentIdentity] = {}
    by_anchor: dict[str, ComponentIdentity] = {}
    enrichment_by_anchor: dict[str, str] = {}
    for component in components:
        identity = canonical_component_identity(
            component, repo_root=repo_root, repository=repository
        )
        enrichment_fingerprint = _component_enrichment_fingerprint(component)
        by_identity[identity.key] = identity
        anchor = component_action_key(
            component, repo_root=repo_root, repository=repository
        )
        # In the unlikely event of an ambiguous fallback anchor, choose the
        # lexicographically first identity/fingerprint pair so results remain
        # order-independent.
        previous = by_anchor.get(anchor)
        previous_fingerprint = enrichment_by_anchor.get(anchor, "")
        if previous is None or (identity.key, enrichment_fingerprint) < (
            previous.key,
            previous_fingerprint,
        ):
            by_anchor[anchor] = identity
            enrichment_by_anchor[anchor] = enrichment_fingerprint
    return by_identity, by_anchor, enrichment_by_anchor


def _canonical_enrichment_value(value: Any) -> Any:
    """Convert nested component values into a deterministic JSON shape."""
    if isinstance(value, Enum):
        return _canonical_enrichment_value(value.value)
    if isinstance(value, BaseModel):
        return _canonical_enrichment_value(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_enrichment_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_enrichment_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        canonical_items = [_canonical_enrichment_value(item) for item in value]
        return sorted(
            canonical_items,
            key=lambda item: json.dumps(item, sort_keys=True, default=str),
        )
    if isinstance(value, Path):
        return str(value)
    return value


def _component_enrichment_fingerprint(component: Any) -> str:
    """Fingerprint substantive non-identity fields for action inference.

    Known model defaults make sparse dictionaries compare equally with their
    equivalent Pydantic representation.  Unknown serialized fields remain in
    the payload unless they are identity or agentic bookkeeping fields.
    """
    payload = dict(_COMPONENT_ENRICHMENT_DEFAULTS)
    if isinstance(component, BaseModel):
        raw_component: Mapping[str, Any] = component.model_dump(mode="python")
    elif isinstance(component, Mapping):
        raw_component = component
    else:
        raw_component = {
            field_name: _value(component, field_name, default=default)
            for field_name, default in _COMPONENT_ENRICHMENT_DEFAULTS.items()
        }

    excluded_fields = _COMPONENT_IDENTITY_FIELDS | _COMPONENT_BOOKKEEPING_FIELDS
    payload.update(
        {
            str(field_name): value
            for field_name, value in raw_component.items()
            if str(field_name) not in excluded_fields
            and not str(field_name).startswith("agentic_")
        }
    )

    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        payload["metadata"] = {
            str(key): value
            for key, value in metadata.items()
            if str(key)
            not in (_METADATA_BOOKKEEPING_FIELDS | _COMPONENT_BOOKKEEPING_FIELDS)
            and not str(key).startswith("agentic_")
        }

    return json.dumps(
        _canonical_enrichment_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _relationship_keys(
    relationships: Iterable[Any],
    *,
    components: Iterable[Any],
    repo_root: str | Path | None,
    repository: str | None,
    present: bool = True,
    expected_labels: bool = False,
) -> set[str]:
    component_list = list(components)
    return {
        canonical_relationship_identity(
            relationship,
            components=component_list,
            repo_root=repo_root,
            repository=repository,
        ).key
        for relationship in relationships
        if _relationship_is_present(relationship, expected_label=expected_labels)
        is present
    }


def _relationship_is_present(relationship: Any, *, expected_label: bool) -> bool:
    presence_fields = (
        ("expected_present", "present", "is_present")
        if expected_label
        else ("predicted_present", "present", "is_present")
    )
    raw = _value(relationship, *presence_fields, default=True)
    if isinstance(raw, str):
        normalized = raw.strip().casefold()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        raise ValueError(f"Unsupported relationship presence value {raw!r}")
    return bool(raw)


def _risk_is_present(risk: Any, *, expected_label: bool) -> bool:
    presence_fields = (
        ("expected_present", "present", "is_present")
        if expected_label
        else ("predicted_present", "present", "is_present")
    )
    raw = _value(risk, *presence_fields, default=True)
    if isinstance(raw, str):
        normalized = raw.strip().casefold()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        raise ValueError(f"Unsupported risk presence value {raw!r}")
    return bool(raw)


def _risk_keys(
    risks: Iterable[Any],
    *,
    repo_root: str | Path | None,
    repository: str | None,
    present: bool = True,
    expected_labels: bool = False,
) -> set[str]:
    return {
        canonical_risk_identity(risk, repo_root=repo_root, repository=repository).key
        for risk in risks
        if _risk_is_present(risk, expected_label=expected_labels) is present
    }


def _decision_for_target(
    baseline: ComponentIdentity,
    target: ComponentIdentity | None,
    *,
    baseline_enrichment: str,
    target_enrichment: str | None,
) -> ActionDecision:
    if target is None:
        return ActionDecision(action="remove")
    if target.component_type != baseline.component_type:
        return ActionDecision(action="reclassify", target_type=target.component_type)
    if target.key != baseline.key or target_enrichment != baseline_enrichment:
        return ActionDecision(action="enrich")
    return ActionDecision(action="keep")


def _infer_actions(
    baseline_by_anchor: Mapping[str, ComponentIdentity],
    target_by_anchor: Mapping[str, ComponentIdentity],
    baseline_enrichment_by_anchor: Mapping[str, str],
    target_enrichment_by_anchor: Mapping[str, str],
) -> dict[str, ActionDecision]:
    actions = {
        anchor: _decision_for_target(
            baseline,
            target_by_anchor.get(anchor),
            baseline_enrichment=baseline_enrichment_by_anchor[anchor],
            target_enrichment=target_enrichment_by_anchor.get(anchor),
        )
        for anchor, baseline in sorted(baseline_by_anchor.items())
    }
    for anchor in sorted(set(target_by_anchor) - set(baseline_by_anchor)):
        actions[anchor] = ActionDecision(action="discover")
    return actions


def _normalise_action_key(raw: Any) -> str:
    key = str(raw).strip()
    if key.startswith(("case:", "loc:")):
        return key
    return f"case:{key}"


def _action_decision(raw: Any) -> ActionDecision:
    if isinstance(raw, ActionDecision):
        return raw
    if isinstance(raw, str):
        action = raw.strip().casefold()
        target_type = ""
    else:
        action = _normalise_token(_value(raw, "action", "decision", default=""))
        target_type = _normalise_token(
            _value(raw, "target_type", "new_type", default="")
        )
    if action not in _ACTION_VALUES:
        raise ValueError(
            f"Unsupported decision action {action!r}; expected one of "
            f"{sorted(_ACTION_VALUES)}"
        )
    return ActionDecision(action=cast(_DecisionAction, action), target_type=target_type)


def _normalise_actions(actions: Any) -> dict[str, ActionDecision]:
    if actions is None:
        return {}
    result: dict[str, ActionDecision] = {}
    if isinstance(actions, Mapping):
        for raw_key, raw_decision in actions.items():
            result[_normalise_action_key(raw_key)] = _action_decision(raw_decision)
        return result
    for item in actions:
        raw_key = _value(item, *_CASE_ID_FIELDS, "action_key", "id")
        if raw_key is None or not str(raw_key).strip():
            raise ValueError(
                "Each action record must define a stable case id/action_key"
            )
        result[_normalise_action_key(raw_key)] = _action_decision(item)
    return result


def _accuracy(
    predicted: Mapping[str, ActionDecision],
    expected: Mapping[str, ActionDecision],
    *,
    reclassifications_only: bool = False,
) -> AccuracyMetric | None:
    expected_items = {
        key: decision
        for key, decision in expected.items()
        if not reclassifications_only or decision.action == "reclassify"
    }
    if not expected_items:
        return None
    correct = 0
    mismatches: list[ActionMismatch] = []
    for key in sorted(expected_items):
        expected_decision = expected_items[key]
        predicted_decision = predicted.get(key)
        is_match = predicted_decision is not None and (
            predicted_decision.action == expected_decision.action
        )
        if (
            is_match
            and predicted_decision is not None
            and expected_decision.action == "reclassify"
            and expected_decision.target_type
        ):
            is_match = predicted_decision.target_type == expected_decision.target_type
        if is_match:
            correct += 1
        else:
            mismatches.append(
                ActionMismatch(
                    action_key=key,
                    expected=expected_decision,
                    predicted=predicted_decision,
                )
            )
    unexpected = sorted(set(predicted) - set(expected))
    evaluated = len(expected_items)
    return AccuracyMetric(
        correct_count=correct,
        evaluated_count=evaluated,
        accuracy=correct / evaluated,
        mismatches=mismatches,
        unexpected_action_keys=unexpected,
    )


def _action_macro_f1_and_coverage(
    predicted: Mapping[str, ActionDecision],
    expected: Mapping[str, ActionDecision],
) -> tuple[float | None, float | None]:
    """Score action classes while treating missing predictions as abstentions.

    Only expected action keys form the evaluation population.  Unexpected keys
    remain available on :class:`AccuracyMetric` but do not create unlabeled
    classification samples.  A missing prediction contributes a false negative
    to its expected class and reduces coverage; it is never imputed as removal.
    Reclassification target type remains part of exact action accuracy and its
    dedicated metric, while macro-F1 evaluates the action class itself.
    """
    if not expected:
        return None, None

    expected_keys = set(expected)
    covered_keys = expected_keys & set(predicted)
    coverage = len(covered_keys) / len(expected_keys)
    classes = {decision.action for decision in expected.values()}
    classes.update(predicted[key].action for key in covered_keys)

    class_f1: list[float] = []
    for action in sorted(classes):
        true_positives = sum(
            expected[key].action == action
            and predicted.get(key) is not None
            and predicted[key].action == action
            for key in expected_keys
        )
        false_positives = sum(
            expected[key].action != action
            and predicted.get(key) is not None
            and predicted[key].action == action
            for key in expected_keys
        )
        false_negatives = sum(
            expected[key].action == action
            and (predicted.get(key) is None or predicted[key].action != action)
            for key in expected_keys
        )
        precision = (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if true_positives + false_negatives
            else 0.0
        )
        class_f1.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return sum(class_f1) / len(class_f1), coverage


def evaluate_decisions(
    *,
    predicted_components: Iterable[Any],
    expected_components: Iterable[Any],
    predicted_relationships: Iterable[Any] = (),
    expected_relationships: Iterable[Any] = (),
    predicted_risks: Iterable[Any] = (),
    expected_risks: Iterable[Any] = (),
    deterministic_components: Iterable[Any] | None = None,
    deterministic_relationships: Iterable[Any] | None = None,
    predicted_actions: Any = None,
    expected_actions: Any = None,
    repo_root: str | Path | None = None,
    repository: str | None = None,
) -> DecisionEvaluationResult:
    """Evaluate exact entity-level AIBOM decisions.

    Parameters
    ----------
    predicted_components, expected_components:
        Final AIBOM output and labelled ground truth.
    deterministic_components:
        Optional pre-agent candidate set.  Supplying it enables baseline recall,
        net recall lift, discovery, over-pruning, and inferred action metrics.
    deterministic_relationships:
        Optional pre-agent relationship set. Supplying it enables exact baseline
        relationship metrics and final-minus-baseline relationship recall lift.
    predicted_relationships, expected_relationships:
        Final and labelled directed edge sets. Relationship labels may set
        ``expected_present=False``/``present=False`` for explicit negatives.
    predicted_risks, expected_risks:
        Final and labelled risk sets. Risk labels may set
        ``expected_present=False``/``present=False`` for explicit negatives.
    predicted_actions, expected_actions:
        Optional explicit action mappings/records. Mapping keys are stable case
        ids (``"case-17"`` or ``"case:case-17"``); values are action strings or
        objects with ``action`` and optional ``target_type``/``new_type``. These
        override inferred before/after actions when supplied.
    repo_root:
        Scan root used to canonicalize absolute paths to repo-relative paths.
    repository:
        Optional stable logical repository label for multi-repository datasets.
    """
    predicted_list = list(predicted_components)
    expected_list = list(expected_components)
    predicted_relationship_list = list(predicted_relationships)
    expected_relationship_list = list(expected_relationships)
    predicted_risk_list = list(predicted_risks)
    expected_risk_list = list(expected_risks)
    baseline_list = (
        list(deterministic_components) if deterministic_components is not None else None
    )
    baseline_relationship_list = (
        list(deterministic_relationships)
        if deterministic_relationships is not None
        else None
    )

    (
        predicted_by_id,
        predicted_by_anchor,
        predicted_enrichment_by_anchor,
    ) = _identity_and_anchor_maps(
        predicted_list, repo_root=repo_root, repository=repository
    )
    (
        expected_by_id,
        expected_by_anchor,
        expected_enrichment_by_anchor,
    ) = _identity_and_anchor_maps(
        expected_list, repo_root=repo_root, repository=repository
    )
    component_metrics, component_details = _compare_sets(
        set(predicted_by_id), set(expected_by_id)
    )

    predicted_relation_keys = _relationship_keys(
        predicted_relationship_list,
        components=predicted_list,
        repo_root=repo_root,
        repository=repository,
    )
    expected_relation_keys = _relationship_keys(
        expected_relationship_list,
        components=expected_list,
        repo_root=repo_root,
        repository=repository,
        expected_labels=True,
    )
    relationship_metrics, relationship_details = _compare_sets(
        predicted_relation_keys, expected_relation_keys
    )

    predicted_risk_keys = _risk_keys(
        predicted_risk_list,
        repo_root=repo_root,
        repository=repository,
    )
    expected_risk_keys = _risk_keys(
        expected_risk_list,
        repo_root=repo_root,
        repository=repository,
        expected_labels=True,
    )
    risk_metrics, risk_details = _compare_sets(predicted_risk_keys, expected_risk_keys)

    baseline_metrics: PRF1Metric | None = None
    baseline_details: SetComparison | None = None
    baseline_relationship_metrics: PRF1Metric | None = None
    baseline_relationship_details: SetComparison | None = None
    net_recall_lift: float | None = None
    relationship_recall_lift: float | None = None
    discovery_metrics: PRF1Metric | None = None
    discovery_details: SetComparison | None = None
    over_pruning: OverPruneMetric | None = None
    inferred_predicted_actions: dict[str, ActionDecision] = {}
    inferred_expected_actions: dict[str, ActionDecision] = {}

    if baseline_relationship_list is not None:
        baseline_relationship_components = (
            baseline_list if baseline_list is not None else expected_list
        )
        baseline_relation_keys = _relationship_keys(
            baseline_relationship_list,
            components=baseline_relationship_components,
            repo_root=repo_root,
            repository=repository,
        )
        (
            baseline_relationship_metrics,
            baseline_relationship_details,
        ) = _compare_sets(baseline_relation_keys, expected_relation_keys)
        relationship_recall_lift = (
            relationship_metrics.recall - baseline_relationship_metrics.recall
        )

    if baseline_list is not None:
        (
            baseline_by_id,
            baseline_by_anchor,
            baseline_enrichment_by_anchor,
        ) = _identity_and_anchor_maps(
            baseline_list, repo_root=repo_root, repository=repository
        )
        baseline_metrics, baseline_details = _compare_sets(
            set(baseline_by_id), set(expected_by_id)
        )
        net_recall_lift = component_metrics.recall - baseline_metrics.recall

        baseline_anchors = set(baseline_by_anchor)
        predicted_discovery_ids = {
            identity.key
            for anchor, identity in predicted_by_anchor.items()
            if anchor not in baseline_anchors
        }
        expected_discovery_ids = {
            identity.key
            for anchor, identity in expected_by_anchor.items()
            if anchor not in baseline_anchors
        }
        discovery_metrics, discovery_details = _compare_sets(
            predicted_discovery_ids, expected_discovery_ids
        )

        eligible = baseline_anchors & set(expected_by_anchor)
        over_pruned = eligible - set(predicted_by_anchor)
        over_pruning = OverPruneMetric(
            over_pruned_count=len(over_pruned),
            eligible_baseline_count=len(eligible),
            rate=len(over_pruned) / len(eligible) if eligible else 0.0,
            over_pruned_action_keys=sorted(over_pruned),
        )

        inferred_predicted_actions = _infer_actions(
            baseline_by_anchor,
            predicted_by_anchor,
            baseline_enrichment_by_anchor,
            predicted_enrichment_by_anchor,
        )
        inferred_expected_actions = _infer_actions(
            baseline_by_anchor,
            expected_by_anchor,
            baseline_enrichment_by_anchor,
            expected_enrichment_by_anchor,
        )

    normalized_predicted_actions = (
        _normalise_actions(predicted_actions)
        if predicted_actions is not None
        else inferred_predicted_actions
    )
    normalized_expected_actions = (
        _normalise_actions(expected_actions)
        if expected_actions is not None
        else inferred_expected_actions
    )
    action_accuracy = _accuracy(
        normalized_predicted_actions, normalized_expected_actions
    )
    reclassification_accuracy = _accuracy(
        normalized_predicted_actions,
        normalized_expected_actions,
        reclassifications_only=True,
    )
    action_macro_f1, decision_coverage = _action_macro_f1_and_coverage(
        normalized_predicted_actions, normalized_expected_actions
    )

    return DecisionEvaluationResult(
        components=component_metrics,
        relationships=relationship_metrics,
        risks=risk_metrics,
        baseline_components=baseline_metrics,
        baseline_relationships=baseline_relationship_metrics,
        net_recall_lift=net_recall_lift,
        relationship_recall_lift=relationship_recall_lift,
        discoveries=discovery_metrics,
        over_pruning=over_pruning,
        action_accuracy=action_accuracy,
        action_macro_f1=action_macro_f1,
        decision_coverage=decision_coverage,
        reclassification_accuracy=reclassification_accuracy,
        details=DecisionEvaluationDetails(
            components=component_details,
            relationships=relationship_details,
            risks=risk_details,
            baseline_components=baseline_details,
            baseline_relationships=baseline_relationship_details,
            discoveries=discovery_details,
        ),
    )


__all__ = [
    "AccuracyMetric",
    "ActionDecision",
    "ActionMismatch",
    "ComponentIdentity",
    "DecisionEvaluationDetails",
    "DecisionEvaluationResult",
    "OverPruneMetric",
    "PRF1Metric",
    "RelationshipIdentity",
    "RiskIdentity",
    "SetComparison",
    "canonical_component_identity",
    "canonical_relationship_identity",
    "canonical_risk_identity",
    "component_action_key",
    "evaluate_decisions",
]
