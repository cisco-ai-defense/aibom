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

from __future__ import annotations
from enum import Enum
from typing import Callable, Literal

from pydantic import BaseModel, Field

from .model_pinning import is_model_pinned
from .models.enums import AIComponentType, RelationshipType
from .models.scan import AIComponent, ComponentRelationship, ScanResult

ComplianceStatus = Literal["pass", "fail", "not_applicable"]


class ComplianceFramework(str, Enum):
    EU_AI_ACT = "EU_AI_ACT"
    OWASP_AGENTIC = "OWASP_AGENTIC"
    NIST_AI_RMF = "NIST_AI_RMF"


class ComplianceRequirement(BaseModel):
    id: str
    framework: ComplianceFramework
    title: str
    description: str
    applicable_types: list[AIComponentType] = Field(default_factory=list)
    check_fn_name: str


class ComplianceCheckResult(BaseModel):
    requirement_id: str
    framework: ComplianceFramework
    title: str
    status: ComplianceStatus
    message: str
    affected_components: list[str] = Field(default_factory=list)


class ComplianceReport(BaseModel):
    framework: ComplianceFramework
    results: list[ComplianceCheckResult] = Field(default_factory=list)
    coverage_pct: float = 0.0
    summary: dict = Field(default_factory=dict)


_SECRET_HIGH_CONFIDENCE = 0.8


def _by_id(components: list[AIComponent]) -> dict[str, AIComponent]:
    return {c.instance_id: c for c in components}


def check_eu_model_transparency(scan: ScanResult, req: ComplianceRequirement) -> ComplianceCheckResult:
    models = [c for c in scan.all_components if c.component_type.is_model_related]
    if not models:
        return ComplianceCheckResult(
            requirement_id=req.id,
            framework=req.framework,
            title=req.title,
            status="not_applicable",
            message="No model components present.",
        )
    bad = [c for c in models if not (c.model_name and str(c.model_name).strip())]
    if bad:
        return ComplianceCheckResult(
            requirement_id=req.id,
            framework=req.framework,
            title=req.title,
            status="fail",
            message="One or more models lack a non-empty model_name.",
            affected_components=[c.name for c in bad],
        )
    return ComplianceCheckResult(
        requirement_id=req.id,
        framework=req.framework,
        title=req.title,
        status="pass",
        message="All models declare model_name.",
    )


def check_eu_training_dataset(scan: ScanResult, req: ComplianceRequirement) -> ComplianceCheckResult:
    has_run = any(c.component_type == AIComponentType.TRAINING_RUN for c in scan.all_components)
    if not has_run:
        return ComplianceCheckResult(
            requirement_id=req.id,
            framework=req.framework,
            title=req.title,
            status="not_applicable",
            message="No training_run components present.",
        )
    has_ds = any(c.component_type == AIComponentType.DATASET for c in scan.all_components)
    if not has_ds:
        return ComplianceCheckResult(
            requirement_id=req.id,
            framework=req.framework,
            title=req.title,
            status="fail",
            message="training_run present but no dataset documented.",
        )
    return ComplianceCheckResult(
        requirement_id=req.id,
        framework=req.framework,
        title=req.title,
        status="pass",
        message="At least one dataset is documented alongside training.",
    )


def check_risk_assessed(scan: ScanResult, req: ComplianceRequirement) -> ComplianceCheckResult:
    assessed = (
        scan.risk.score > 0
        or bool(scan.risk.flags)
        or bool(scan.metadata.get("risk_assessed"))
    )
    if assessed:
        return ComplianceCheckResult(
            requirement_id=req.id,
            framework=req.framework,
            title=req.title,
            status="pass",
            message="Risk score or explicit assessment recorded.",
        )
    return ComplianceCheckResult(
        requirement_id=req.id,
        framework=req.framework,
        title=req.title,
        status="fail",
        message="No risk contribution or explicit assessment (score is 0 and no flags).",
    )


def check_eu_agent_guardrails(scan: ScanResult, req: ComplianceRequirement) -> ComplianceCheckResult:
    agents = [c for c in scan.all_components if c.component_type == AIComponentType.AGENT]
    if not agents:
        return ComplianceCheckResult(
            requirement_id=req.id,
            framework=req.framework,
            title=req.title,
            status="not_applicable",
            message="No agent components present.",
        )
    rels = scan.all_relationships
    bad: list[AIComponent] = []
    for a in agents:
        guarded = any(
            r.relationship_type == RelationshipType.USES_GUARDRAIL
            and r.source_instance_id == a.instance_id
            for r in rels
        )
        if not guarded:
            bad.append(a)
    if bad:
        return ComplianceCheckResult(
            requirement_id=req.id,
            framework=req.framework,
            title=req.title,
            status="fail",
            message="One or more agents lack a USES_GUARDRAIL relationship.",
            affected_components=[c.name for c in bad],
        )
    return ComplianceCheckResult(
        requirement_id=req.id,
        framework=req.framework,
        title=req.title,
        status="pass",
        message="All agents are linked to guardrails.",
    )


def check_eu_no_agentic_leftover(scan: ScanResult, req: ComplianceRequirement) -> ComplianceCheckResult:
    pending = [c for c in scan.all_components if c.needs_agentic]
    if pending:
        return ComplianceCheckResult(
            requirement_id=req.id,
            framework=req.framework,
            title=req.title,
            status="fail",
            message="Components still require agentic identification.",
            affected_components=[c.name for c in pending],
        )
    return ComplianceCheckResult(
        requirement_id=req.id,
        framework=req.framework,
        title=req.title,
        status="pass",
        message="No components flagged needs_agentic.",
    )


def check_owasp_agent_inventory(scan: ScanResult, req: ComplianceRequirement) -> ComplianceCheckResult:
    agents = [c for c in scan.all_components if c.component_type == AIComponentType.AGENT]
    if not agents:
        return ComplianceCheckResult(
            requirement_id=req.id,
            framework=req.framework,
            title=req.title,
            status="not_applicable",
            message="No agent components present.",
        )
    bad = [
        c for c in agents
        if not ((c.description and str(c.description).strip()) or (c.framework and str(c.framework).strip()))
    ]
    if bad:
        return ComplianceCheckResult(
            requirement_id=req.id,
            framework=req.framework,
            title=req.title,
            status="fail",
            message="Agents must include description or framework metadata.",
            affected_components=[c.name for c in bad],
        )
    return ComplianceCheckResult(
        requirement_id=req.id,
        framework=req.framework,
        title=req.title,
        status="pass",
        message="All agents have description or framework set.",
    )


def check_owasp_tool_documentation(scan: ScanResult, req: ComplianceRequirement) -> ComplianceCheckResult:
    tools = [c for c in scan.all_components if c.component_type == AIComponentType.TOOL]
    if not tools:
        return ComplianceCheckResult(
            requirement_id=req.id,
            framework=req.framework,
            title=req.title,
            status="not_applicable",
            message="No tool components present.",
        )
    rels = scan.all_relationships
    bad: list[AIComponent] = []
    for t in tools:
        documented = any(
            r.relationship_type == RelationshipType.USES_TOOL
            and r.target_instance_id == t.instance_id
            for r in rels
        )
        if not documented:
            bad.append(t)
    if bad:
        return ComplianceCheckResult(
            requirement_id=req.id,
            framework=req.framework,
            title=req.title,
            status="fail",
            message="Tools should be referenced by USES_TOOL from an agent.",
            affected_components=[c.name for c in bad],
        )
    return ComplianceCheckResult(
        requirement_id=req.id,
        framework=req.framework,
        title=req.title,
        status="pass",
        message="All tools are linked from agents.",
    )


def check_owasp_mcp_guardrails(scan: ScanResult, req: ComplianceRequirement) -> ComplianceCheckResult:
    servers = [c for c in scan.all_components if c.component_type == AIComponentType.MCP_SERVER]
    if not servers:
        return ComplianceCheckResult(
            requirement_id=req.id,
            framework=req.framework,
            title=req.title,
            status="not_applicable",
            message="No MCP server components present.",
        )
    rels = scan.all_relationships
    comp_map = _by_id(scan.all_components)
    bad: list[AIComponent] = []
    for s in servers:
        ok = False
        for r in rels:
            if r.relationship_type != RelationshipType.USES_GUARDRAIL:
                continue
            if r.source_instance_id != s.instance_id and r.target_instance_id != s.instance_id:
                continue
            peer_id = r.target_instance_id if r.source_instance_id == s.instance_id else r.source_instance_id
            peer = comp_map.get(peer_id)
            if peer and peer.component_type == AIComponentType.GUARDRAIL:
                ok = True
                break
        if not ok:
            bad.append(s)
    if bad:
        return ComplianceCheckResult(
            requirement_id=req.id,
            framework=req.framework,
            title=req.title,
            status="fail",
            message="MCP servers should be associated with guardrails via USES_GUARDRAIL.",
            affected_components=[c.name for c in bad],
        )
    return ComplianceCheckResult(
        requirement_id=req.id,
        framework=req.framework,
        title=req.title,
        status="pass",
        message="All MCP servers are linked to guardrails.",
    )


def check_owasp_secret_management(scan: ScanResult, req: ComplianceRequirement) -> ComplianceCheckResult:
    secrets = [
        c for c in scan.all_components
        if c.component_type == AIComponentType.SECRET and c.confidence >= _SECRET_HIGH_CONFIDENCE
    ]
    if not secrets:
        return ComplianceCheckResult(
            requirement_id=req.id,
            framework=req.framework,
            title=req.title,
            status="not_applicable",
            message="No high-confidence secret detections.",
        )
    return ComplianceCheckResult(
        requirement_id=req.id,
        framework=req.framework,
        title=req.title,
        status="fail",
        message="High-confidence secret findings indicate possible hardcoded secrets.",
        affected_components=[c.name for c in secrets],
    )


def check_owasp_agent_observability(scan: ScanResult, req: ComplianceRequirement) -> ComplianceCheckResult:
    agents = [c for c in scan.all_components if c.component_type == AIComponentType.AGENT]
    if not agents:
        return ComplianceCheckResult(
            requirement_id=req.id,
            framework=req.framework,
            title=req.title,
            status="not_applicable",
            message="No agent components present.",
        )
    rels = scan.all_relationships
    bad: list[AIComponent] = []
    for a in agents:
        observes = any(
            r.relationship_type == RelationshipType.OBSERVES and r.source_instance_id == a.instance_id
            for r in rels
        )
        if not observes:
            bad.append(a)
    if bad:
        return ComplianceCheckResult(
            requirement_id=req.id,
            framework=req.framework,
            title=req.title,
            status="fail",
            message="Agents should emit OBSERVES relationships for observability.",
            affected_components=[c.name for c in bad],
        )
    return ComplianceCheckResult(
        requirement_id=req.id,
        framework=req.framework,
        title=req.title,
        status="pass",
        message="All agents have observability relationships.",
    )


def check_nist_asset_inventory(scan: ScanResult, req: ComplianceRequirement) -> ComplianceCheckResult:
    ai_types = frozenset(t for t in AIComponentType if t != AIComponentType.OTHER)
    found = any(c.component_type in ai_types for c in scan.all_components)
    if found:
        return ComplianceCheckResult(
            requirement_id=req.id,
            framework=req.framework,
            title=req.title,
            status="pass",
            message="At least one non-other AI component detected.",
        )
    return ComplianceCheckResult(
        requirement_id=req.id,
        framework=req.framework,
        title=req.title,
        status="fail",
        message="No AI-type components detected in scan.",
    )


def check_nist_model_pinned(scan: ScanResult, req: ComplianceRequirement) -> ComplianceCheckResult:
    models = [c for c in scan.all_components if c.component_type.is_model_related]
    if not models:
        return ComplianceCheckResult(
            requirement_id=req.id,
            framework=req.framework,
            title=req.title,
            status="not_applicable",
            message="No model components present.",
        )
    bad: list[AIComponent] = []
    for m in models:
        mn = (m.model_name or m.name or "").strip()
        if not is_model_pinned(mn):
            bad.append(m)
    if bad:
        return ComplianceCheckResult(
            requirement_id=req.id,
            framework=req.framework,
            title=req.title,
            status="fail",
            message="Models should pin versions (e.g. semver, org/model id, or tag).",
            affected_components=[c.name for c in bad],
        )
    return ComplianceCheckResult(
        requirement_id=req.id,
        framework=req.framework,
        title=req.title,
        status="pass",
        message="Model identifiers include version or registry path cues.",
    )


def check_nist_dataset_provenance(scan: ScanResult, req: ComplianceRequirement) -> ComplianceCheckResult:
    datasets = [c for c in scan.all_components if c.component_type == AIComponentType.DATASET]
    if not datasets:
        return ComplianceCheckResult(
            requirement_id=req.id,
            framework=req.framework,
            title=req.title,
            status="not_applicable",
            message="No dataset components present.",
        )
    bad = [c for c in datasets if not (c.dataset_source and str(c.dataset_source).strip())]
    if bad:
        return ComplianceCheckResult(
            requirement_id=req.id,
            framework=req.framework,
            title=req.title,
            status="fail",
            message="Datasets should declare dataset_source for provenance.",
            affected_components=[c.name for c in bad],
        )
    return ComplianceCheckResult(
        requirement_id=req.id,
        framework=req.framework,
        title=req.title,
        status="pass",
        message="All datasets declare provenance.",
    )


def check_nist_monitoring(scan: ScanResult, req: ComplianceRequirement) -> ComplianceCheckResult:
    agents = [c for c in scan.all_components if c.component_type == AIComponentType.AGENT]
    if not agents:
        return ComplianceCheckResult(
            requirement_id=req.id,
            framework=req.framework,
            title=req.title,
            status="not_applicable",
            message="No agent components present.",
        )
    has_obs = any(c.component_type == AIComponentType.OBSERVABILITY for c in scan.all_components)
    if has_obs:
        return ComplianceCheckResult(
            requirement_id=req.id,
            framework=req.framework,
            title=req.title,
            status="pass",
            message="Observability components present alongside agents.",
        )
    return ComplianceCheckResult(
        requirement_id=req.id,
        framework=req.framework,
        title=req.title,
        status="fail",
        message="Agents present but no observability components detected.",
    )


_CHECK_REGISTRY: dict[str, Callable[[ScanResult, ComplianceRequirement], ComplianceCheckResult]] = {
    "check_eu_model_transparency": check_eu_model_transparency,
    "check_eu_training_dataset": check_eu_training_dataset,
    "check_risk_assessed": check_risk_assessed,
    "check_eu_agent_guardrails": check_eu_agent_guardrails,
    "check_eu_no_agentic_leftover": check_eu_no_agentic_leftover,
    "check_owasp_agent_inventory": check_owasp_agent_inventory,
    "check_owasp_tool_documentation": check_owasp_tool_documentation,
    "check_owasp_mcp_guardrails": check_owasp_mcp_guardrails,
    "check_owasp_secret_management": check_owasp_secret_management,
    "check_owasp_agent_observability": check_owasp_agent_observability,
    "check_nist_asset_inventory": check_nist_asset_inventory,
    "check_nist_model_pinned": check_nist_model_pinned,
    "check_nist_dataset_provenance": check_nist_dataset_provenance,
    "check_nist_monitoring": check_nist_monitoring,
}

BUILTIN_REQUIREMENTS: list[ComplianceRequirement] = [
    ComplianceRequirement(
        id="eu-1",
        framework=ComplianceFramework.EU_AI_ACT,
        title="Model transparency",
        description="Every model must have model_name set.",
        applicable_types=[AIComponentType.MODEL, AIComponentType.LLM_ENDPOINT, AIComponentType.MODEL_ENDPOINT],
        check_fn_name="check_eu_model_transparency",
    ),
    ComplianceRequirement(
        id="eu-2",
        framework=ComplianceFramework.EU_AI_ACT,
        title="Dataset documentation",
        description="If training_run exists, at least one dataset should exist.",
        applicable_types=[AIComponentType.TRAINING_RUN, AIComponentType.DATASET],
        check_fn_name="check_eu_training_dataset",
    ),
    ComplianceRequirement(
        id="eu-3",
        framework=ComplianceFramework.EU_AI_ACT,
        title="Risk assessment",
        description="Risk score must be computed or explicitly assessed.",
        applicable_types=[],
        check_fn_name="check_risk_assessed",
    ),
    ComplianceRequirement(
        id="eu-4",
        framework=ComplianceFramework.EU_AI_ACT,
        title="Human oversight",
        description="Agents must be linked to guardrails via USES_GUARDRAIL.",
        applicable_types=[AIComponentType.AGENT, AIComponentType.GUARDRAIL],
        check_fn_name="check_eu_agent_guardrails",
    ),
    ComplianceRequirement(
        id="eu-5",
        framework=ComplianceFramework.EU_AI_ACT,
        title="Technical documentation",
        description="All component types should be identified (no needs_agentic remaining).",
        applicable_types=[],
        check_fn_name="check_eu_no_agentic_leftover",
    ),
    ComplianceRequirement(
        id="owasp-1",
        framework=ComplianceFramework.OWASP_AGENTIC,
        title="Agent inventory",
        description="Agents must have description or framework set.",
        applicable_types=[AIComponentType.AGENT],
        check_fn_name="check_owasp_agent_inventory",
    ),
    ComplianceRequirement(
        id="owasp-2",
        framework=ComplianceFramework.OWASP_AGENTIC,
        title="Tool authorization",
        description="Tools used by agents should be documented via relationships.",
        applicable_types=[AIComponentType.TOOL],
        check_fn_name="check_owasp_tool_documentation",
    ),
    ComplianceRequirement(
        id="owasp-3",
        framework=ComplianceFramework.OWASP_AGENTIC,
        title="MCP server security",
        description="MCP servers should have associated guardrails.",
        applicable_types=[AIComponentType.MCP_SERVER],
        check_fn_name="check_owasp_mcp_guardrails",
    ),
    ComplianceRequirement(
        id="owasp-4",
        framework=ComplianceFramework.OWASP_AGENTIC,
        title="Secret management",
        description="No high-confidence hardcoded secret detections.",
        applicable_types=[AIComponentType.SECRET],
        check_fn_name="check_owasp_secret_management",
    ),
    ComplianceRequirement(
        id="owasp-5",
        framework=ComplianceFramework.OWASP_AGENTIC,
        title="Observability",
        description="Agents should have OBSERVES relationships.",
        applicable_types=[AIComponentType.AGENT],
        check_fn_name="check_owasp_agent_observability",
    ),
    ComplianceRequirement(
        id="nist-1",
        framework=ComplianceFramework.NIST_AI_RMF,
        title="AI asset inventory",
        description="At least one AI-type component should be detected.",
        applicable_types=[],
        check_fn_name="check_nist_asset_inventory",
    ),
    ComplianceRequirement(
        id="nist-2",
        framework=ComplianceFramework.NIST_AI_RMF,
        title="Model governance",
        description="Models should be pinned (version or registry path).",
        applicable_types=[AIComponentType.MODEL, AIComponentType.LLM_ENDPOINT, AIComponentType.MODEL_ENDPOINT],
        check_fn_name="check_nist_model_pinned",
    ),
    ComplianceRequirement(
        id="nist-3",
        framework=ComplianceFramework.NIST_AI_RMF,
        title="Data governance",
        description="Datasets should declare dataset_source.",
        applicable_types=[AIComponentType.DATASET],
        check_fn_name="check_nist_dataset_provenance",
    ),
    ComplianceRequirement(
        id="nist-4",
        framework=ComplianceFramework.NIST_AI_RMF,
        title="Monitoring",
        description="Observability components should exist when agents are present.",
        applicable_types=[AIComponentType.AGENT, AIComponentType.OBSERVABILITY],
        check_fn_name="check_nist_monitoring",
    ),
    ComplianceRequirement(
        id="nist-5",
        framework=ComplianceFramework.NIST_AI_RMF,
        title="Risk management",
        description="Risk assessment has been performed.",
        applicable_types=[],
        check_fn_name="check_risk_assessed",
    ),
]


def evaluate_compliance(scan_result: ScanResult, framework: ComplianceFramework) -> ComplianceReport:
    reqs = [r for r in BUILTIN_REQUIREMENTS if r.framework == framework]
    results: list[ComplianceCheckResult] = []
    for req in reqs:
        fn = _CHECK_REGISTRY[req.check_fn_name]
        results.append(fn(scan_result, req))
    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    na = sum(1 for r in results if r.status == "not_applicable")
    denom = passed + failed
    coverage = (passed / denom * 100.0) if denom else 100.0
    summary = {
        "total_requirements": len(results),
        "passed": passed,
        "failed": failed,
        "not_applicable": na,
        "coverage_pct": coverage,
    }
    return ComplianceReport(
        framework=framework,
        results=results,
        coverage_pct=coverage,
        summary=summary,
    )


def parse_compliance_cli_value(value: str) -> ComplianceFramework | Literal["all"]:
    v = value.strip().lower().replace("_", "-")
    mapping = {
        "eu-ai-act": ComplianceFramework.EU_AI_ACT,
        "owasp-agentic": ComplianceFramework.OWASP_AGENTIC,
        "nist-ai-rmf": ComplianceFramework.NIST_AI_RMF,
    }
    if v == "all":
        return "all"
    if v in mapping:
        return mapping[v]
    raise ValueError(f"Unknown compliance framework: {value!r}")
