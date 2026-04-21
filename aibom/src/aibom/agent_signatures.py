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

"""Built-in catalog of agent-framework, protocol, anti-pattern, and
remote-agent-SDK signatures.

These signatures are the data layer the evidence builder and the
:mod:`aibom.scanners.remote_agent_resolver` match against per-class
observations (imports, base classes, control-flow, calls, and
protocol-relevant string literals).

Design
------
* **Data, not code.** Each entry is a plain dataclass literal so the
  builder has zero framework-specific conditional branches.
* **One config file.** The built-in catalog lives in *this* Python module,
  not in a shipped YAML. Users extend/override it via an optional
  ``agent_signatures:`` section in their existing ``.aibom.yaml``.
* **Evidence-bearing.** Every framework/protocol signature declares which
  :class:`aibom.agentic.agent.AgentEvidence` ``pattern`` value it should
  produce, so the downstream prompt and verification gate see a typed,
  bounded claim.
* **Anti-patterns are first-class.** Matching anti-patterns preclude
  agent classification regardless of what else matches.
* **Remote agent SDKs are distinct from protocols.** A
  :class:`RemoteAgentSdkSignature` targets SDKs where the *server* runs
  the agent loop (OpenAI Assistants, LangGraph Cloud, AWS Bedrock
  Agents Runtime, AWS Bedrock AgentCore Runtime, Vertex Reasoning
  Engines). Plain LLM-completion SDKs deliberately do NOT appear here
  — they do not change the local class's role.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signature dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentFrameworkSignature:
    """A named framework API whose presence is positive evidence of agency.

    A signature matches a class when any of the following hit:

    * ``entrypoint_qualified_names`` matches a call inside the class or a
      constructor assignment whose RHS is the entrypoint.
    * ``base_class_names`` matches one of the class's declared bases
      (short name or qualified name).
    * ``import_substrings`` narrows attribution — only matches if the file
      that defines the class imports one of the substrings. Used to
      disambiguate short names like ``Agent`` that many libraries use.

    ``evidence_pattern`` MUST be one of the :class:`AgentEvidence.pattern`
    literals (``framework_agent``, ``framework_inheritance``, …).
    """

    id: str
    framework: str
    evidence_pattern: str
    entrypoint_qualified_names: tuple[str, ...] = ()
    base_class_names: tuple[str, ...] = ()
    import_substrings: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class AgentProtocolSignature:
    """A protocol-level signal (A2A, MCP, OpenAI Assistants, …).

    Some protocols are *positive* evidence of an agent on their own
    (A2A server, OpenAI Assistants client). Others — notably MCP — are
    tool-provider / tool-consumer signals that by themselves do NOT
    indicate agency. The ``evidence_pattern`` captures this:

    * ``a2a_server`` — class is registered as an A2A agent endpoint.
    * ``remote_proxy`` — class invokes a remote agent. Whether the remote
      end really is an agent is confirmed by Phase 4's remote resolver.
    * ``other`` — informational only (MCP, generic HTTP).
    """

    id: str
    protocol: str
    evidence_pattern: str
    role: str  # "server" | "client" | "unknown"
    import_substrings: tuple[str, ...] = ()
    qualified_name_substrings: tuple[str, ...] = ()
    string_literal_substrings: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class AgentAntiPatternSignature:
    """A signal that a class is *definitively not* an agent.

    Matches are surfaced so the LLM can see the exclusion reason, and the
    Phase 6 verification gate uses them to reject any ``agent`` verdict
    on a class that matches an anti-pattern.
    """

    id: str
    label: str
    base_class_names: tuple[str, ...] = ()
    decorator_qualified_names: tuple[str, ...] = ()
    import_substrings: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class RemoteAgentSdkSignature:
    """A known SDK where *the server* runs the agentic loop.

    Local code that invokes one of these SDK calls is an ``AGENT_PROXY``:
    the remote service owns the control flow, tool dispatch, and
    iteration; the local class is just a client. This is deliberately
    narrower than "any HTTP/SSE call into an LLM endpoint" — plain
    token-streaming chat completions are *not* remote agent proxies
    (the caller runs the loop, not the server).

    A match requires BOTH signals, which together keep false positives
    low without needing ReAct-style heuristics:

    * The file's imports contain at least one substring from
      :attr:`import_substrings` (so we know the file has access to the
      SDK), and
    * At least one call inside a class body whose
      :attr:`~aibom.structures.CallObservation.qualified_name` ends with
      one of :attr:`call_qualified_suffixes`.

    When both hit, :mod:`aibom.scanners.remote_agent_resolver` emits a
    proxy component attributed to the enclosing class. Module-scope
    calls are ignored.

    ``import_substrings`` may be empty, in which case the call-suffix
    signal alone is sufficient (useful for user-defined internal SDKs
    distributed under a local package path).

    The LLM still owns the final classification; this signature just
    raises a structurally-plausible candidate so the agentic stage sees
    it in the prompt.
    """

    id: str
    sdk_name: str
    import_substrings: tuple[str, ...] = ()
    call_qualified_suffixes: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class VerificationPolicy:
    """Tuning knobs for the evidence-builder and verification gate.

    ``min_react_loop_distinct_callees`` was bumped from 2 to 3 after an
    e2e review found that two-callee loops were routinely HTTP pollers,
    retry wrappers, and ETL activities rather than ReAct-style
    orchestration. A ReAct loop has at minimum: read state → call LLM →
    dispatch tool, which gives three distinct callees naturally.
    """

    require_evidence_for_agent: bool = True
    allow_remote_proxy_without_cross_repo: bool = False
    min_react_loop_call_count: int = 2
    min_react_loop_distinct_callees: int = 3


@dataclass
class AgentSignatureCatalog:
    """Full signature set used by the evidence builder."""

    frameworks: list[AgentFrameworkSignature] = field(default_factory=list)
    protocols: list[AgentProtocolSignature] = field(default_factory=list)
    anti_patterns: list[AgentAntiPatternSignature] = field(default_factory=list)
    remote_agent_sdks: list[RemoteAgentSdkSignature] = field(default_factory=list)
    verification_policy: VerificationPolicy = field(default_factory=VerificationPolicy)

    @property
    def is_empty(self) -> bool:
        return (
            not self.frameworks
            and not self.protocols
            and not self.anti_patterns
            and not self.remote_agent_sdks
            and self.verification_policy == VerificationPolicy()
        )


# ---------------------------------------------------------------------------
# Built-in defaults (package-shipped, Python literals, not YAML)
# ---------------------------------------------------------------------------

_BUILTIN_FRAMEWORKS: tuple[AgentFrameworkSignature, ...] = (
    AgentFrameworkSignature(
        id="langchain.AgentExecutor",
        framework="langchain",
        evidence_pattern="framework_agent",
        entrypoint_qualified_names=(
            "langchain.agents.AgentExecutor",
            "langchain.agents.AgentExecutor.from_agent_and_tools",
            "langchain_core.agents.AgentExecutor",
            "AgentExecutor",
        ),
        import_substrings=("langchain.agents", "langchain_core.agents"),
        description="LangChain AgentExecutor — canonical ReAct orchestration loop.",
    ),
    AgentFrameworkSignature(
        id="langchain.BaseSingleActionAgent",
        framework="langchain",
        evidence_pattern="framework_inheritance",
        base_class_names=(
            "BaseSingleActionAgent",
            "BaseMultiActionAgent",
            "langchain.agents.BaseSingleActionAgent",
            "langchain.agents.BaseMultiActionAgent",
        ),
        import_substrings=("langchain.agents",),
        description="Subclass of LangChain's canonical agent base classes.",
    ),
    AgentFrameworkSignature(
        id="langgraph.create_react_agent",
        framework="langgraph",
        evidence_pattern="framework_agent",
        entrypoint_qualified_names=(
            "langgraph.prebuilt.create_react_agent",
            "create_react_agent",
        ),
        import_substrings=("langgraph.prebuilt", "langgraph"),
        description="LangGraph prebuilt ReAct agent factory.",
    ),
    AgentFrameworkSignature(
        id="autogen.AssistantAgent",
        framework="autogen",
        evidence_pattern="framework_agent",
        entrypoint_qualified_names=(
            "autogen.AssistantAgent",
            "autogen.UserProxyAgent",
            "autogen_agentchat.agents.AssistantAgent",
            "AssistantAgent",
            "UserProxyAgent",
        ),
        import_substrings=("autogen", "autogen_agentchat"),
        description="AutoGen AssistantAgent / UserProxyAgent multi-turn chat agents.",
    ),
    AgentFrameworkSignature(
        id="crewai.Agent",
        framework="crewai",
        evidence_pattern="framework_agent",
        entrypoint_qualified_names=(
            "crewai.Agent",
            "Agent",  # disambiguated by import_substrings
        ),
        import_substrings=("crewai",),
        description="CrewAI Agent constructor.",
    ),
    AgentFrameworkSignature(
        id="llama_index.ReActAgent",
        framework="llama_index",
        evidence_pattern="framework_agent",
        entrypoint_qualified_names=(
            "llama_index.core.agent.ReActAgent",
            "llama_index.agent.ReActAgent",
            "ReActAgent",
        ),
        import_substrings=("llama_index",),
        description="LlamaIndex ReActAgent.",
    ),
    AgentFrameworkSignature(
        id="llama_index.BaseAgent",
        framework="llama_index",
        evidence_pattern="framework_inheritance",
        base_class_names=(
            "BaseAgent",
            "llama_index.core.agent.BaseAgent",
        ),
        import_substrings=("llama_index",),
        description="Subclass of LlamaIndex BaseAgent.",
    ),
    AgentFrameworkSignature(
        id="openai_agents.Agent",
        framework="openai_agents",
        evidence_pattern="framework_agent",
        entrypoint_qualified_names=("agents.Agent",),
        import_substrings=("agents.run", "agents.agent"),
        description="OpenAI Agents SDK `agents.Agent`.",
    ),
    AgentFrameworkSignature(
        id="pydantic_ai.Agent",
        framework="pydantic_ai",
        evidence_pattern="framework_agent",
        entrypoint_qualified_names=("pydantic_ai.Agent",),
        import_substrings=("pydantic_ai",),
        description="Pydantic AI Agent.",
    ),
    AgentFrameworkSignature(
        id="haystack.Agent",
        framework="haystack",
        evidence_pattern="framework_agent",
        entrypoint_qualified_names=(
            "haystack.agents.Agent",
            "haystack_experimental.core.Agent",
        ),
        import_substrings=("haystack.agents", "haystack_experimental"),
        description="Haystack Agent.",
    ),
    AgentFrameworkSignature(
        id="strands.Agent",
        framework="strands",
        evidence_pattern="framework_agent",
        entrypoint_qualified_names=(
            "strands.Agent",
            "strands.agent.Agent",
            "Agent",
        ),
        import_substrings=("strands",),
        description="AWS Strands Agent — the canonical top-level agent class.",
    ),
    AgentFrameworkSignature(
        id="strands.experimental.BidiAgent",
        framework="strands",
        evidence_pattern="framework_agent",
        entrypoint_qualified_names=(
            "strands.experimental.bidi.BidiAgent",
            "strands.experimental.BidiAgent",
            "BidiAgent",
        ),
        import_substrings=("strands.experimental.bidi", "strands.experimental"),
        description="Strands BidiAgent (experimental bidirectional streaming agent).",
    ),
)


_BUILTIN_PROTOCOLS: tuple[AgentProtocolSignature, ...] = (
    AgentProtocolSignature(
        id="a2a.server",
        protocol="a2a",
        evidence_pattern="a2a_server",
        role="server",
        import_substrings=("a2a.server", "a2a_sdk"),
        qualified_name_substrings=("A2AServer",),
        string_literal_substrings=(".well-known/agent.json",),
        description="Exposes an A2A endpoint; serves an Agent Card.",
    ),
    AgentProtocolSignature(
        id="a2a.client",
        protocol="a2a",
        evidence_pattern="remote_proxy",
        role="client",
        import_substrings=("a2a.client",),
        qualified_name_substrings=("A2AClient",),
        string_literal_substrings=("message/send", "message/stream", "tasks/get"),
        description=(
            "Calls a remote A2A agent. The remote side must be independently "
            "verified (Phase 4) before this class is classified as an agent."
        ),
    ),
    # MCP is a tool/resource protocol. It is NOT evidence of agency by itself;
    # an MCP server is a tool provider, an MCP client is a tool consumer. The
    # evidence_pattern is "other" so the LLM sees it as informational context.
    AgentProtocolSignature(
        id="mcp.server",
        protocol="mcp",
        evidence_pattern="other",
        role="server",
        import_substrings=("mcp.server", "modelcontextprotocol"),
        qualified_name_substrings=("FastMCP",),
        string_literal_substrings=("tools/list", "resources/list", "prompts/list"),
        description=(
            "MCP server — exposes tools/resources/prompts to clients. "
            "By itself this is NOT evidence of an agent."
        ),
    ),
    AgentProtocolSignature(
        id="mcp.client",
        protocol="mcp",
        evidence_pattern="other",
        role="client",
        import_substrings=("mcp.client",),
        qualified_name_substrings=("ClientSession",),
        string_literal_substrings=("tools/call",),
        description=(
            "MCP client — consumes tools from an MCP server. "
            "By itself this is NOT evidence of an agent."
        ),
    ),
    AgentProtocolSignature(
        id="openai.assistants",
        protocol="openai_assistants",
        evidence_pattern="remote_proxy",
        role="client",
        qualified_name_substrings=(
            "client.beta.assistants.create",
            "client.beta.threads.runs.create",
            "openai.beta.assistants",
            "openai.beta.threads.runs",
        ),
        string_literal_substrings=("/v1/assistants", "/v1/threads"),
        description="Invokes a remote OpenAI Assistant — server-side agent.",
    ),
)


_BUILTIN_ANTI_PATTERNS: tuple[AgentAntiPatternSignature, ...] = (
    AgentAntiPatternSignature(
        id="temporal.workflow",
        label="temporal_workflow",
        decorator_qualified_names=(
            "temporalio.workflow.defn",
            "workflow.defn",
        ),
        import_substrings=("temporalio.workflow",),
        description="Temporal workflow — deterministic replay, not an agent.",
    ),
    AgentAntiPatternSignature(
        id="temporal.activity",
        label="temporal_activity",
        decorator_qualified_names=(
            "temporalio.activity.defn",
            "activity.defn",
        ),
        import_substrings=("temporalio.activity",),
        description="Temporal activity — deterministic task, not an agent.",
    ),
    AgentAntiPatternSignature(
        id="celery.task",
        label="celery_task",
        decorator_qualified_names=(
            "celery.app.task.Task",
            "celery.Celery.task",
            "app.task",
            "celery_app.task",
        ),
        import_substrings=("celery",),
        description="Celery task — async job, not an agent.",
    ),
    AgentAntiPatternSignature(
        id="airflow.dag",
        label="airflow_dag",
        base_class_names=(
            "BaseOperator",
            "airflow.models.BaseOperator",
            "airflow.operators.python.PythonOperator",
        ),
        decorator_qualified_names=("airflow.decorators.dag", "dag"),
        import_substrings=("airflow",),
        description="Airflow DAG/Operator — scheduled batch, not an agent.",
    ),
    AgentAntiPatternSignature(
        id="fastapi.endpoint",
        label="fastapi_endpoint",
        decorator_qualified_names=(
            "fastapi.APIRouter.get",
            "fastapi.APIRouter.post",
            "app.get",
            "app.post",
            "router.get",
            "router.post",
            "router.put",
            "router.delete",
            "router.patch",
        ),
        import_substrings=("fastapi",),
        description="FastAPI HTTP endpoint — may call an agent but is not itself one.",
    ),
    AgentAntiPatternSignature(
        id="pydantic.basemodel",
        label="pydantic_basemodel",
        base_class_names=("BaseModel", "pydantic.BaseModel"),
        import_substrings=("pydantic",),
        description="Pure Pydantic data class — not an agent.",
    ),
)


# Remote agent-runtime SDKs — the server runs the agent loop.
#
# These are used by :mod:`aibom.scanners.remote_agent_resolver` to emit
# ``AGENT_PROXY`` components from call sites that are too structurally
# plain to resemble an A2A client. The patterns are intentionally
# narrow: they target SDKs for which we have strong public documentation
# that the *remote* service executes a tool-using agent loop, not
# generic "LLM over HTTP" SDKs.
_BUILTIN_REMOTE_AGENT_SDKS: tuple[RemoteAgentSdkSignature, ...] = (
    RemoteAgentSdkSignature(
        id="openai.assistants",
        sdk_name="OpenAI Assistants API",
        import_substrings=("openai",),
        call_qualified_suffixes=(
            ".threads.runs.create",
            ".threads.runs.create_and_stream",
            ".threads.runs.create_and_poll",
            ".threads.runs.stream",
            ".threads.runs.poll",
            ".threads.runs.submit_tool_outputs",
            ".threads.runs.submit_tool_outputs_stream",
        ),
        description=(
            "OpenAI Assistants API — the server runs the tool-using "
            "agent loop, the local client is a proxy. Detected only via "
            "calls to ``client.beta.threads.runs.*`` inside a class "
            "whose file imports ``openai``; plain ``chat.completions`` "
            "token streaming deliberately does NOT match."
        ),
    ),
    RemoteAgentSdkSignature(
        id="langgraph.cloud_sdk",
        sdk_name="LangGraph Cloud / LangGraph Platform SDK",
        import_substrings=("langgraph_sdk",),
        call_qualified_suffixes=(
            ".runs.create",
            ".runs.stream",
            ".runs.wait",
            ".threads.runs.create",
            ".threads.runs.stream",
            ".threads.runs.wait",
        ),
        description=(
            "LangGraph SDK (``langgraph_sdk.get_client``) targeting a "
            "deployed LangGraph Cloud / Platform server that runs the "
            "graph. Detected via ``client.runs.*`` / "
            "``client.threads.runs.*`` calls in a file that imports "
            "``langgraph_sdk``."
        ),
    ),
    RemoteAgentSdkSignature(
        id="langgraph.remote_graph",
        sdk_name="LangGraph RemoteGraph",
        import_substrings=("langgraph.pregel.remote",),
        call_qualified_suffixes=(
            "RemoteGraph",
        ),
        description=(
            "LangGraph ``RemoteGraph`` — proxies to a deployed LangGraph "
            "server that executes the orchestration. Detected only when "
            "the file imports ``langgraph.pregel.remote`` so that plain "
            "generic ``.invoke``/``.stream`` attribute calls do not "
            "trip the detector."
        ),
    ),
    RemoteAgentSdkSignature(
        id="aws.bedrock_agents",
        sdk_name="AWS Bedrock Agents Runtime (classic)",
        import_substrings=("boto3", "botocore", "aiobotocore"),
        call_qualified_suffixes=(
            ".invoke_agent",
            ".invoke_agent_with_response_stream",
            ".invoke_inline_agent",
            ".invoke_flow",
        ),
        description=(
            "AWS Bedrock Agents Runtime — the classic ``bedrock-agent-"
            "runtime`` boto3 service hosting AWS-managed agents with "
            "action groups and knowledge bases. The agent loop runs "
            "server-side; the local class is a proxy. Generic "
            "``invoke_model`` (plain LLM completion on the "
            "``bedrock-runtime`` service) is deliberately excluded."
        ),
    ),
    RemoteAgentSdkSignature(
        id="aws.bedrock_agentcore_runtime",
        sdk_name="AWS Bedrock AgentCore Runtime",
        import_substrings=("boto3", "botocore", "aiobotocore"),
        call_qualified_suffixes=(
            ".invoke_agent_runtime",
        ),
        description=(
            "AWS Bedrock AgentCore Runtime — the newer ``bedrock-"
            "agentcore`` boto3 service that hosts customer-built "
            "agents (LangGraph, CrewAI, Strands, custom code) inside "
            "AWS-managed sandboxes. The agent loop executes inside "
            "AgentCore on AWS infrastructure; local code that calls "
            "``client.invoke_agent_runtime(agentRuntimeArn=…, "
            "payload=…)`` is a proxy. This is intentionally separate "
            "from ``aws.bedrock_agents`` because the two services have "
            "different boto3 client names (``bedrock-agentcore`` vs "
            "``bedrock-agent-runtime``), different IAM permissions, "
            "and different operational models."
        ),
    ),
    RemoteAgentSdkSignature(
        id="gcp.vertex_reasoning_engines",
        sdk_name="Vertex AI Reasoning Engines",
        import_substrings=(
            "vertexai.preview.reasoning_engines",
            "vertexai.reasoning_engines",
        ),
        call_qualified_suffixes=(
            "ReasoningEngine",
            ".reasoning_engines.ReasoningEngine",
        ),
        description=(
            "Vertex AI Reasoning Engines — the deployed engine runs the "
            "agent. Local ``ReasoningEngine(...)`` constructions or "
            "bindings delegate control flow to the remote runtime."
        ),
    ),
)


def default_catalog() -> AgentSignatureCatalog:
    """Return a fresh copy of the built-in catalog.

    Defensive copy so callers can mutate / merge without corrupting the
    module-level defaults.
    """
    return AgentSignatureCatalog(
        frameworks=list(_BUILTIN_FRAMEWORKS),
        protocols=list(_BUILTIN_PROTOCOLS),
        anti_patterns=list(_BUILTIN_ANTI_PATTERNS),
        remote_agent_sdks=list(_BUILTIN_REMOTE_AGENT_SDKS),
        verification_policy=VerificationPolicy(),
    )


# ---------------------------------------------------------------------------
# Parse user overrides from .aibom.yaml
# ---------------------------------------------------------------------------


def _as_tuple_of_str(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(v) for v in value if v is not None)
    return ()


def _parse_framework(raw: dict[str, Any], index: int) -> AgentFrameworkSignature | None:
    id_ = raw.get("id")
    pattern = raw.get("evidence_pattern")
    if not id_ or not isinstance(id_, str):
        LOGGER.warning(
            "agent_signatures.frameworks[%d]: 'id' is required and must be a string.",
            index,
        )
        return None
    if not pattern or not isinstance(pattern, str):
        LOGGER.warning(
            "agent_signatures.frameworks[%d] ('%s'): "
            "'evidence_pattern' is required and must be a string.",
            index,
            id_,
        )
        return None
    return AgentFrameworkSignature(
        id=id_,
        framework=str(raw.get("framework", "custom")),
        evidence_pattern=pattern,
        entrypoint_qualified_names=_as_tuple_of_str(raw.get("entrypoint_qualified_names")),
        base_class_names=_as_tuple_of_str(raw.get("base_class_names")),
        import_substrings=_as_tuple_of_str(raw.get("import_substrings")),
        description=str(raw.get("description", "")),
    )


def _parse_protocol(raw: dict[str, Any], index: int) -> AgentProtocolSignature | None:
    id_ = raw.get("id")
    protocol = raw.get("protocol")
    pattern = raw.get("evidence_pattern")
    if not id_ or not isinstance(id_, str):
        LOGGER.warning(
            "agent_signatures.protocols[%d]: 'id' is required and must be a string.",
            index,
        )
        return None
    if not protocol or not isinstance(protocol, str):
        LOGGER.warning(
            "agent_signatures.protocols[%d] ('%s'): 'protocol' is required.",
            index,
            id_,
        )
        return None
    if not pattern or not isinstance(pattern, str):
        LOGGER.warning(
            "agent_signatures.protocols[%d] ('%s'): 'evidence_pattern' is required.",
            index,
            id_,
        )
        return None
    return AgentProtocolSignature(
        id=id_,
        protocol=protocol,
        evidence_pattern=pattern,
        role=str(raw.get("role", "unknown")),
        import_substrings=_as_tuple_of_str(raw.get("import_substrings")),
        qualified_name_substrings=_as_tuple_of_str(raw.get("qualified_name_substrings")),
        string_literal_substrings=_as_tuple_of_str(raw.get("string_literal_substrings")),
        description=str(raw.get("description", "")),
    )


def _parse_anti_pattern(raw: dict[str, Any], index: int) -> AgentAntiPatternSignature | None:
    id_ = raw.get("id")
    label = raw.get("label")
    if not id_ or not isinstance(id_, str):
        LOGGER.warning(
            "agent_signatures.anti_patterns[%d]: 'id' is required.",
            index,
        )
        return None
    if not label or not isinstance(label, str):
        LOGGER.warning(
            "agent_signatures.anti_patterns[%d] ('%s'): 'label' is required.",
            index,
            id_,
        )
        return None
    return AgentAntiPatternSignature(
        id=id_,
        label=label,
        base_class_names=_as_tuple_of_str(raw.get("base_class_names")),
        decorator_qualified_names=_as_tuple_of_str(raw.get("decorator_qualified_names")),
        import_substrings=_as_tuple_of_str(raw.get("import_substrings")),
        description=str(raw.get("description", "")),
    )


def _parse_remote_agent_sdk(
    raw: dict[str, Any], index: int
) -> RemoteAgentSdkSignature | None:
    id_ = raw.get("id")
    if not id_ or not isinstance(id_, str):
        LOGGER.warning(
            "agent_signatures.remote_agent_sdks[%d]: 'id' is required and must be a string.",
            index,
        )
        return None
    call_suffixes = _as_tuple_of_str(raw.get("call_qualified_suffixes"))
    if not call_suffixes:
        LOGGER.warning(
            "agent_signatures.remote_agent_sdks[%d] ('%s'): "
            "'call_qualified_suffixes' is required and must contain at least one entry.",
            index,
            id_,
        )
        return None
    sdk_name = raw.get("sdk_name")
    if not sdk_name or not isinstance(sdk_name, str):
        sdk_name = id_
    return RemoteAgentSdkSignature(
        id=id_,
        sdk_name=sdk_name,
        import_substrings=_as_tuple_of_str(raw.get("import_substrings")),
        call_qualified_suffixes=call_suffixes,
        description=str(raw.get("description", "")),
    )


def _parse_verification_policy(raw: Any) -> VerificationPolicy | None:
    """Parse the ``verification_policy`` sub-mapping.

    Returns ``None`` if *raw* is not a mapping (caller then keeps the
    built-in defaults). Fields that are not provided fall back to the
    built-in defaults field-by-field.
    """
    if not isinstance(raw, dict):
        return None
    defaults = VerificationPolicy()
    return VerificationPolicy(
        require_evidence_for_agent=bool(
            raw.get("require_evidence_for_agent", defaults.require_evidence_for_agent)
        ),
        allow_remote_proxy_without_cross_repo=bool(
            raw.get(
                "allow_remote_proxy_without_cross_repo",
                defaults.allow_remote_proxy_without_cross_repo,
            )
        ),
        min_react_loop_call_count=int(
            raw.get("min_react_loop_call_count", defaults.min_react_loop_call_count)
        ),
        min_react_loop_distinct_callees=int(
            raw.get(
                "min_react_loop_distinct_callees",
                defaults.min_react_loop_distinct_callees,
            )
        ),
    )


def parse_user_signatures(raw: Any) -> AgentSignatureCatalog:
    """Parse the contents of the ``agent_signatures:`` section of .aibom.yaml.

    Returns an empty catalog on malformed input. Sub-entries that fail
    validation are skipped with a warning. The caller merges the returned
    catalog with the built-in defaults via :func:`merge_catalogs`.
    """
    result = AgentSignatureCatalog()
    if not isinstance(raw, dict):
        if raw is not None:
            LOGGER.warning(
                "'agent_signatures' must be a mapping; got %s.", type(raw).__name__
            )
        return result

    known_keys = {
        "frameworks",
        "protocols",
        "anti_patterns",
        "remote_agent_sdks",
        "verification_policy",
    }
    unknown = set(raw.keys()) - known_keys
    for key in sorted(unknown):
        LOGGER.warning(
            "Unknown key 'agent_signatures.%s'. Known keys: %s",
            key,
            ", ".join(sorted(known_keys)),
        )

    raw_frameworks = raw.get("frameworks") or []
    if isinstance(raw_frameworks, list):
        for idx, entry in enumerate(raw_frameworks):
            if isinstance(entry, dict):
                parsed = _parse_framework(entry, idx)
                if parsed is not None:
                    result.frameworks.append(parsed)

    raw_protocols = raw.get("protocols") or []
    if isinstance(raw_protocols, list):
        for idx, entry in enumerate(raw_protocols):
            if isinstance(entry, dict):
                parsed = _parse_protocol(entry, idx)
                if parsed is not None:
                    result.protocols.append(parsed)

    raw_anti = raw.get("anti_patterns") or []
    if isinstance(raw_anti, list):
        for idx, entry in enumerate(raw_anti):
            if isinstance(entry, dict):
                parsed = _parse_anti_pattern(entry, idx)
                if parsed is not None:
                    result.anti_patterns.append(parsed)

    raw_sdks = raw.get("remote_agent_sdks") or []
    if isinstance(raw_sdks, list):
        for idx, entry in enumerate(raw_sdks):
            if isinstance(entry, dict):
                parsed_sdk = _parse_remote_agent_sdk(entry, idx)
                if parsed_sdk is not None:
                    result.remote_agent_sdks.append(parsed_sdk)

    policy = _parse_verification_policy(raw.get("verification_policy"))
    if policy is not None:
        result.verification_policy = policy

    return result


# ---------------------------------------------------------------------------
# Merge built-in + user catalogs
# ---------------------------------------------------------------------------


def _merge_by_id(
    base: Sequence[Any], extra: Sequence[Any]
) -> list[Any]:
    """Merge two id-keyed sequences. Entries in *extra* override entries in
    *base* with the same ``id``; new entries are appended in first-seen order.
    """
    by_id: dict[str, Any] = {}
    for item in base:
        by_id[item.id] = item
    for item in extra:
        by_id[item.id] = item
    return list(by_id.values())


def merge_catalogs(
    builtin: AgentSignatureCatalog,
    user: AgentSignatureCatalog | None,
) -> AgentSignatureCatalog:
    """Merge a user catalog on top of the built-in catalog.

    * For each of the three signature lists, user entries are APPENDED after
      built-ins. If a user entry's ``id`` matches a built-in ``id``, the
      user entry REPLACES the built-in (user wins on id collision).
    * ``verification_policy`` from the user fully replaces the built-in
      when the user parser returned a non-``None`` policy. If the user
      provided no ``verification_policy`` key, the built-in is preserved.
    """
    if user is None or user.is_empty:
        return AgentSignatureCatalog(
            frameworks=list(builtin.frameworks),
            protocols=list(builtin.protocols),
            anti_patterns=list(builtin.anti_patterns),
            remote_agent_sdks=list(builtin.remote_agent_sdks),
            verification_policy=builtin.verification_policy,
        )
    return AgentSignatureCatalog(
        frameworks=_merge_by_id(builtin.frameworks, user.frameworks),
        protocols=_merge_by_id(builtin.protocols, user.protocols),
        anti_patterns=_merge_by_id(builtin.anti_patterns, user.anti_patterns),
        remote_agent_sdks=_merge_by_id(
            builtin.remote_agent_sdks, user.remote_agent_sdks
        ),
        verification_policy=user.verification_policy,
    )


def resolve_catalog(user: AgentSignatureCatalog | None = None) -> AgentSignatureCatalog:
    """Convenience: built-in defaults merged with an optional user catalog."""
    return merge_catalogs(default_catalog(), user)
