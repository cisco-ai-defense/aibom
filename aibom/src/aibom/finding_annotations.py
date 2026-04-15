from __future__ import annotations

from pathlib import Path

from .models import (
    AIComponent,
    CodeSnippet,
    ComponentRelationship,
    DecisionAnnotation,
    EvidenceLocation,
)
from .models.scan import RiskFlag


def _evidence_location(
    file_path: str,
    line_number: int,
    *,
    role: str,
) -> EvidenceLocation | None:
    if not file_path:
        return None
    start_line = max(line_number or 1, 1)
    end_line = start_line
    return EvidenceLocation(
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        role=role,
    )


def _is_within_roots(file_path: str, allowed_roots: list[str]) -> bool:
    """Return True only if *file_path* resolves inside one of *allowed_roots*."""
    if not allowed_roots:
        return False
    try:
        resolved = Path(file_path).resolve()
    except OSError:
        return False
    return any(
        resolved == Path(root).resolve() or Path(root).resolve() in resolved.parents
        for root in allowed_roots
    )


def _read_code_snippet(
    location: EvidenceLocation | None,
    *,
    allowed_roots: list[str] | None = None,
) -> CodeSnippet | None:
    if location is None or not location.file_path:
        return None
    if allowed_roots is not None and not _is_within_roots(location.file_path, allowed_roots):
        return None
    try:
        lines = Path(location.file_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    start_line = max(location.start_line, 1)
    end_line = max(location.end_line, start_line)
    if start_line > len(lines):
        return None

    excerpt_end = min(end_line, start_line + 4, len(lines))
    excerpt = lines[start_line - 1:excerpt_end]
    if not excerpt:
        return None

    return CodeSnippet(
        file_path=location.file_path,
        start_line=start_line,
        end_line=start_line + len(excerpt) - 1,
        text="\n".join(excerpt) + "\n",
        truncated=excerpt_end < end_line or excerpt_end < len(lines),
    )


def _hydrate_annotation(
    annotation: DecisionAnnotation,
    *,
    include_code_snippets: bool,
    snippet_location: EvidenceLocation | None,
    allowed_roots: list[str] | None = None,
) -> DecisionAnnotation:
    if not include_code_snippets or annotation.code_snippet is not None:
        return annotation
    snippet = _read_code_snippet(snippet_location, allowed_roots=allowed_roots)
    if snippet is None:
        return annotation
    return annotation.model_copy(update={"code_snippet": snippet})


def _component_annotation(
    component: AIComponent,
    *,
    include_code_snippets: bool,
    allowed_roots: list[str] | None = None,
) -> DecisionAnnotation:
    primary_location = _evidence_location(
        component.file_path,
        component.line_number,
        role="primary",
    )
    if component.decision_annotation is not None:
        return _hydrate_annotation(
            component.decision_annotation,
            include_code_snippets=include_code_snippets,
            snippet_location=primary_location,
            allowed_roots=allowed_roots,
        )

    evidence_locations = [primary_location] if primary_location is not None else []
    annotation = DecisionAnnotation(
        decision="confirmed",
        justification=(
            f"Kept in the final AIBOM because the scan identified "
            f"{component.component_type.value.replace('_', ' ')} '{component.name}'."
        ),
        evidence_kinds=["code_context"] if evidence_locations else ["scan_result"],
        evidence_locations=evidence_locations,
    )
    return _hydrate_annotation(
        annotation,
        include_code_snippets=include_code_snippets,
        snippet_location=primary_location,
        allowed_roots=allowed_roots,
    )


def _relationship_annotation(
    relationship: ComponentRelationship,
    *,
    component_by_id: dict[str, AIComponent],
    include_code_snippets: bool,
    allowed_roots: list[str] | None = None,
) -> DecisionAnnotation:
    source = component_by_id.get(relationship.source_instance_id)
    target = component_by_id.get(relationship.target_instance_id)
    evidence_locations = [
        location
        for location in [
            _evidence_location(
                source.file_path if source else "",
                source.line_number if source else 0,
                role="source",
            ),
            _evidence_location(
                target.file_path if target else "",
                target.line_number if target else 0,
                role="target",
            ),
        ]
        if location is not None
    ]
    primary_location = evidence_locations[0] if evidence_locations else None
    if relationship.decision_annotation is not None:
        return _hydrate_annotation(
            relationship.decision_annotation,
            include_code_snippets=include_code_snippets,
            snippet_location=primary_location,
            allowed_roots=allowed_roots,
        )

    source_label = relationship.source_name or relationship.source_instance_id or "source"
    target_label = relationship.target_name or relationship.target_instance_id or "target"
    annotation = DecisionAnnotation(
        decision="derived",
        justification=(
            f"Kept in the final AIBOM because the scan linked "
            f"{source_label} to {target_label} via "
            f"{relationship.relationship_type.value.replace('_', ' ').lower()}."
        ),
        evidence_kinds=["relationship_context"],
        evidence_locations=evidence_locations,
    )
    return _hydrate_annotation(
        annotation,
        include_code_snippets=include_code_snippets,
        snippet_location=primary_location,
        allowed_roots=allowed_roots,
    )


def _risk_annotation(
    flag: RiskFlag,
    *,
    include_code_snippets: bool,
    allowed_roots: list[str] | None = None,
) -> DecisionAnnotation:
    primary_location = _evidence_location(
        flag.file_path,
        flag.line_number,
        role="primary",
    )
    if flag.decision_annotation is not None:
        return _hydrate_annotation(
            flag.decision_annotation,
            include_code_snippets=include_code_snippets,
            snippet_location=primary_location,
            allowed_roots=allowed_roots,
        )

    evidence_locations = [primary_location] if primary_location is not None else []
    annotation = DecisionAnnotation(
        decision="flagged",
        justification=flag.description or f"Flagged because the {flag.flag} rule matched.",
        evidence_kinds=["code_context"] if evidence_locations else ["risk_rule"],
        evidence_locations=evidence_locations,
    )
    return _hydrate_annotation(
        annotation,
        include_code_snippets=include_code_snippets,
        snippet_location=primary_location,
        allowed_roots=allowed_roots,
    )


def annotate_findings(
    components: list[AIComponent],
    relationships: list[ComponentRelationship],
    risk_flags: list[RiskFlag],
    *,
    include_code_snippets: bool = False,
    allowed_roots: list[str] | None = None,
) -> tuple[list[AIComponent], list[ComponentRelationship], list[RiskFlag]]:
    """Ensure every final finding carries a decision annotation."""
    annotated_components = [
        component.model_copy(
            update={
                "decision_annotation": _component_annotation(
                    component,
                    include_code_snippets=include_code_snippets,
                    allowed_roots=allowed_roots,
                )
            }
        )
        for component in components
    ]
    component_by_id = {component.instance_id: component for component in annotated_components}
    annotated_relationships = [
        relationship.model_copy(
            update={
                "decision_annotation": _relationship_annotation(
                    relationship,
                    component_by_id=component_by_id,
                    include_code_snippets=include_code_snippets,
                    allowed_roots=allowed_roots,
                )
            }
        )
        for relationship in relationships
    ]
    annotated_risk_flags = [
        flag.model_copy(
            update={
                "decision_annotation": _risk_annotation(
                    flag,
                    include_code_snippets=include_code_snippets,
                    allowed_roots=allowed_roots,
                )
            }
        )
        for flag in risk_flags
    ]
    return annotated_components, annotated_relationships, annotated_risk_flags
