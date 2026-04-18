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

"""End-to-end tests for the agent-evidence-builder scanner.

Each test drives synthetic Python source through the libcst parser, then
through :func:`build_dossiers`, and asserts on the resulting dossier
shape. This validates both the Phase 1 observation emission and the
Phase 2 catalog matching together.
"""

from __future__ import annotations

import textwrap

import pytest

from aibom.agent_signatures import (
    AgentAntiPatternSignature,
    AgentFrameworkSignature,
    AgentProtocolSignature,
    AgentSignatureCatalog,
    VerificationPolicy,
    default_catalog,
)
from aibom.cst_parser import parse_source_code
from aibom.scanners.agent_evidence_builder import (
    AgentEvidenceDossier,
    build_dossier_for_class,
    build_dossiers,
    render_dossier_for_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(source: str, file_path: str = "test.py"):
    return parse_source_code(file_path, textwrap.dedent(source))


def _dossier_for(
    source: str,
    class_name: str,
    catalog: AgentSignatureCatalog | None = None,
    file_path: str = "test.py",
) -> AgentEvidenceDossier:
    result = _parse(source, file_path=file_path)
    cat = catalog if catalog is not None else default_catalog()
    dossiers = build_dossiers(cat, [result])
    matching = [d for d in dossiers if d.class_name == class_name]
    assert matching, (
        f"no dossier produced for class '{class_name}'. "
        f"Observed classes: {[d.class_name for d in dossiers]}"
    )
    return matching[0]


# ---------------------------------------------------------------------------
# Framework matches
# ---------------------------------------------------------------------------


class TestFrameworkMatches:
    def test_langchain_agent_executor_entrypoint_matches(self) -> None:
        source = """
        from langchain.agents import AgentExecutor, create_react_agent

        class MyOrchestrator:
            def build(self, tools, llm):
                return AgentExecutor.from_agent_and_tools(
                    agent=create_react_agent(llm, tools),
                    tools=tools,
                )
        """
        dossier = _dossier_for(source, "MyOrchestrator")
        assert dossier.has_direct_agent_evidence is True
        framework_ids = {m.signature_id for m in dossier.framework_matches}
        assert "langchain.AgentExecutor" in framework_ids
        assert dossier.preferred_pattern in ("framework_agent", "framework_inheritance")

    def test_langchain_inheritance_matches(self) -> None:
        source = """
        from langchain.agents import BaseSingleActionAgent

        class MyAgent(BaseSingleActionAgent):
            def plan(self, intermediate_steps, **kwargs):
                return None
        """
        dossier = _dossier_for(source, "MyAgent")
        assert dossier.has_direct_agent_evidence is True
        patterns = {m.evidence_pattern for m in dossier.framework_matches}
        assert "framework_inheritance" in patterns

    def test_langgraph_create_react_agent(self) -> None:
        source = """
        from langgraph.prebuilt import create_react_agent

        class Router:
            def make(self, llm, tools):
                self.agent = create_react_agent(llm, tools)
                return self.agent
        """
        dossier = _dossier_for(source, "Router")
        assert dossier.has_direct_agent_evidence is True
        ids = {m.signature_id for m in dossier.framework_matches}
        assert "langgraph.create_react_agent" in ids

    def test_crewai_agent_requires_import(self) -> None:
        """Short name ``Agent`` only matches when the file imports crewai."""
        source_without_import = """
        class Demo:
            def build(self):
                return Agent(role="helper")
        """
        dossier = _dossier_for(source_without_import, "Demo")
        crewai_matches = [
            m for m in dossier.framework_matches if m.signature_id == "crewai.Agent"
        ]
        assert crewai_matches == [], (
            "crewai.Agent must not match without a crewai import"
        )

        source_with_import = """
        from crewai import Agent

        class Demo:
            def build(self):
                return Agent(role="helper")
        """
        dossier = _dossier_for(source_with_import, "Demo")
        crewai_matches = [
            m for m in dossier.framework_matches if m.signature_id == "crewai.Agent"
        ]
        assert len(crewai_matches) == 1

    def test_non_agent_class_has_no_framework_match(self) -> None:
        source = """
        class DataContainer:
            value: int
            name: str
        """
        dossier = _dossier_for(source, "DataContainer")
        assert dossier.framework_matches == []
        assert dossier.has_direct_agent_evidence is False


# ---------------------------------------------------------------------------
# Protocol matches
# ---------------------------------------------------------------------------


class TestProtocolMatches:
    def test_a2a_server_string_literal_is_positive_agent_evidence(self) -> None:
        source = """
        class MyServer:
            def register(self):
                self.card_path = "/.well-known/agent.json"
                return self.card_path
        """
        dossier = _dossier_for(source, "MyServer")
        a2a_matches = [
            m for m in dossier.protocol_matches if m.signature_id == "a2a.server"
        ]
        assert len(a2a_matches) == 1
        assert a2a_matches[0].evidence_pattern == "a2a_server"
        # a2a_server counts as direct agent evidence
        assert dossier.has_direct_agent_evidence is True

    def test_a2a_client_is_remote_proxy_not_direct_agent(self) -> None:
        source = """
        from a2a.client import A2AClient

        class RemoteCaller:
            def invoke(self, client: A2AClient, msg):
                client.send("message/send", msg)
                return "ok"
        """
        dossier = _dossier_for(source, "RemoteCaller")
        a2a_matches = [
            m for m in dossier.protocol_matches if m.signature_id == "a2a.client"
        ]
        assert a2a_matches, "a2a.client signature should have matched"
        assert dossier.has_remote_proxy_evidence is True
        # remote_proxy alone is NOT direct agent evidence (needs Phase 4 confirmation).
        assert dossier.has_direct_agent_evidence is False

    def test_mcp_server_does_not_imply_agency(self) -> None:
        source = """
        from mcp.server import FastMCP

        class MyToolProvider:
            def __init__(self):
                self.server = FastMCP()

            def list_tools(self):
                return "tools/list"
        """
        dossier = _dossier_for(source, "MyToolProvider")
        mcp_matches = [
            m for m in dossier.protocol_matches if m.signature_id == "mcp.server"
        ]
        assert mcp_matches, "mcp.server signature should have matched"
        for m in mcp_matches:
            assert m.evidence_pattern == "other"
        # MCP alone is not direct agent evidence
        assert dossier.has_direct_agent_evidence is False

    def test_mcp_client_does_not_imply_agency(self) -> None:
        source = """
        from mcp.client import ClientSession

        class MyToolConsumer:
            def use(self, session: ClientSession):
                return session.call("tools/call", {})
        """
        dossier = _dossier_for(source, "MyToolConsumer")
        mcp_matches = [
            m for m in dossier.protocol_matches if m.signature_id == "mcp.client"
        ]
        assert mcp_matches
        assert dossier.has_direct_agent_evidence is False

    def test_openai_assistants_is_remote_proxy(self) -> None:
        source = """
        import openai

        class AssistantCaller:
            def run(self, client, thread_id, assistant_id):
                run = client.beta.threads.runs.create(
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                )
                return run
        """
        dossier = _dossier_for(source, "AssistantCaller")
        oa_matches = [
            m for m in dossier.protocol_matches if m.signature_id == "openai.assistants"
        ]
        assert oa_matches, "openai.assistants should have matched on runs.create"
        assert oa_matches[0].evidence_pattern == "remote_proxy"
        assert dossier.has_remote_proxy_evidence is True


# ---------------------------------------------------------------------------
# Anti-pattern matches
# ---------------------------------------------------------------------------


class TestAntiPatternMatches:
    def test_temporal_workflow_anti_pattern(self) -> None:
        source = """
        from temporalio import workflow

        @workflow.defn
        class MyWorkflow:
            @workflow.run
            async def run(self):
                return "done"
        """
        dossier = _dossier_for(source, "MyWorkflow")
        labels = {m.label for m in dossier.anti_pattern_matches}
        assert "temporal_workflow" in labels
        assert dossier.is_excluded_by_anti_pattern is True

    def test_celery_task_anti_pattern(self) -> None:
        source = """
        from celery import Celery

        app = Celery("app")

        class Jobs:
            @app.task
            def do_it(self):
                return "done"
        """
        dossier = _dossier_for(source, "Jobs")
        labels = {m.label for m in dossier.anti_pattern_matches}
        assert "celery_task" in labels

    def test_airflow_dag_anti_pattern_via_base(self) -> None:
        source = """
        from airflow.models import BaseOperator

        class MyOp(BaseOperator):
            def execute(self, context):
                return None
        """
        dossier = _dossier_for(source, "MyOp")
        labels = {m.label for m in dossier.anti_pattern_matches}
        assert "airflow_dag" in labels

    def test_fastapi_endpoint_anti_pattern(self) -> None:
        source = """
        from fastapi import APIRouter

        router = APIRouter()

        class Routes:
            @router.get("/agents")
            def list_agents(self):
                return []
        """
        dossier = _dossier_for(source, "Routes")
        labels = {m.label for m in dossier.anti_pattern_matches}
        assert "fastapi_endpoint" in labels

    def test_pydantic_basemodel_anti_pattern(self) -> None:
        source = """
        from pydantic import BaseModel

        class SearchRequest(BaseModel):
            query: str
            limit: int = 10
        """
        dossier = _dossier_for(source, "SearchRequest")
        labels = {m.label for m in dossier.anti_pattern_matches}
        assert "pydantic_basemodel" in labels

    def test_anti_pattern_requires_import(self) -> None:
        """A class with a FastAPI-looking decorator must not match if the
        file does not import fastapi.
        """
        source = """
        class Routes:
            @router.get("/agents")
            def list_agents(self):
                return []
        """
        dossier = _dossier_for(source, "Routes")
        labels = {m.label for m in dossier.anti_pattern_matches}
        assert "fastapi_endpoint" not in labels


# ---------------------------------------------------------------------------
# ReAct-loop detection
# ---------------------------------------------------------------------------


class TestReactLoopDetection:
    def test_react_loop_matches_when_structural_requirements_met(self) -> None:
        source = """
        class MyAgent:
            def run(self, llm, tools, prompt):
                response = None
                for _ in range(10):
                    response = llm.invoke(prompt)
                    if response.tool_calls:
                        result = tools.execute(response.tool_calls[0])
                        prompt = prompt.extend(result)
                    else:
                        return response
                return response
        """
        dossier = _dossier_for(source, "MyAgent")
        assert dossier.react_loop_matches, (
            "ReAct loop should match: 3 distinct callees (llm.invoke, "
            "tools.execute, prompt.extend), a branch, and an LLM-like "
            "call name"
        )
        match = dossier.react_loop_matches[0]
        assert match.evidence_pattern == "react_loop"
        assert match.signature_id == "structural.react_loop"
        # The loop has an LLM-like call name, which should be reflected
        assert "LLM-like" in match.rationale

    def test_react_loop_rejected_without_branch(self) -> None:
        source = """
        class PlainLoop:
            def run(self, llm, tools):
                for _ in range(10):
                    x = llm.invoke("hi")
                    y = tools.execute(x)
                return y
        """
        dossier = _dossier_for(source, "PlainLoop")
        assert dossier.react_loop_matches == [], (
            "No branch in loop body, so not a ReAct match"
        )

    def test_react_loop_rejected_with_insufficient_distinct_callees(self) -> None:
        source = """
        class OneCall:
            def run(self, llm):
                for _ in range(10):
                    response = llm.invoke("hi")
                    if response == "stop":
                        return response
                return None
        """
        dossier = _dossier_for(source, "OneCall")
        # Only ``llm.invoke`` is a distinct call, and
        # min_react_loop_distinct_callees defaults to 3, so this loop
        # must not match regardless of the policy bump history.
        assert dossier.react_loop_matches == []

    def test_react_loop_policy_override_via_catalog(self) -> None:
        """Relaxing ``min_react_loop_distinct_callees`` to 1 must accept a
        loop with a single distinct call.
        """
        source = """
        class Relaxed:
            def run(self, llm):
                for _ in range(10):
                    response = llm.invoke("hi")
                    llm.invoke("again")
                    if response == "stop":
                        return response
                return None
        """
        relaxed_policy = VerificationPolicy(
            min_react_loop_call_count=2,
            min_react_loop_distinct_callees=1,
        )
        catalog = AgentSignatureCatalog(
            frameworks=[],
            protocols=[],
            anti_patterns=[],
            verification_policy=relaxed_policy,
        )
        dossier = _dossier_for(source, "Relaxed", catalog=catalog)
        assert dossier.react_loop_matches, (
            "Relaxed policy must accept a single-callee branching loop"
        )


# ---------------------------------------------------------------------------
# Dossier-level properties and helpers
# ---------------------------------------------------------------------------


class TestDossierProperties:
    def test_body_source_is_captured(self) -> None:
        source = """
        class MyClass:
            x: int = 0

            def hello(self):
                return self.x
        """
        dossier = _dossier_for(source, "MyClass")
        assert "class MyClass" in dossier.class_body_source
        assert "def hello" in dossier.class_body_source

    def test_preferred_pattern_priority(self) -> None:
        """When both framework and A2A server match, framework wins
        (it's a stronger claim than a protocol endpoint)."""
        source = """
        from langchain.agents import AgentExecutor

        class Dual:
            card_path = "/.well-known/agent.json"
            def run(self, llm, tools):
                return AgentExecutor.from_agent_and_tools(
                    agent=None,
                    tools=tools,
                )
        """
        dossier = _dossier_for(source, "Dual")
        assert dossier.framework_matches
        assert dossier.protocol_matches
        assert dossier.preferred_pattern == "framework_agent"

    def test_has_direct_evidence_false_for_pure_proxy(self) -> None:
        source = """
        from a2a.client import A2AClient

        class Client:
            def go(self, c: A2AClient):
                c.send("message/send")
        """
        dossier = _dossier_for(source, "Client")
        assert dossier.has_direct_agent_evidence is False
        assert dossier.has_remote_proxy_evidence is True

    def test_render_dossier_for_prompt_shape(self) -> None:
        source = """
        from langchain.agents import AgentExecutor

        class Demo:
            def go(self, tools):
                return AgentExecutor.from_agent_and_tools(tools=tools, agent=None)
        """
        dossier = _dossier_for(source, "Demo")
        rendered = render_dossier_for_prompt(dossier)
        assert rendered["class_name"] == "Demo"
        assert rendered["preferred_pattern"] in (
            "framework_agent",
            "framework_inheritance",
        )
        assert rendered["has_direct_agent_evidence"] is True
        assert isinstance(rendered["framework_matches"], list)
        assert "pattern" in rendered["framework_matches"][0]
        assert "rationale" in rendered["framework_matches"][0]


# ---------------------------------------------------------------------------
# User-extension interaction
# ---------------------------------------------------------------------------


class TestUserExtensionIntegration:
    def test_user_framework_signature_matches(self) -> None:
        source = """
        from myorg.agents import MyInternalAgent

        class WrappedAgent(MyInternalAgent):
            def step(self):
                return "ok"
        """
        user_catalog = AgentSignatureCatalog(
            frameworks=[
                AgentFrameworkSignature(
                    id="myorg.MyInternalAgent",
                    framework="myorg",
                    evidence_pattern="framework_inheritance",
                    base_class_names=("MyInternalAgent",),
                    import_substrings=("myorg.agents",),
                )
            ]
        )
        # Use the user-only catalog (no built-in LangChain etc.) to isolate.
        result = _parse(source)
        dossier = build_dossier_for_class(
            user_catalog,
            result,
            next(c for c in result.class_bodies if c.class_name == "WrappedAgent"),
        )
        assert dossier.framework_matches
        assert dossier.framework_matches[0].signature_id == "myorg.MyInternalAgent"

    def test_user_anti_pattern_excludes(self) -> None:
        source = """
        from myorg.jobs import ScheduledJob

        class NightlyJob(ScheduledJob):
            def execute(self):
                return "done"
        """
        user_catalog = AgentSignatureCatalog(
            anti_patterns=[
                AgentAntiPatternSignature(
                    id="myorg.scheduled_job",
                    label="scheduled_job",
                    base_class_names=("ScheduledJob",),
                    import_substrings=("myorg.jobs",),
                )
            ]
        )
        result = _parse(source)
        dossier = build_dossier_for_class(
            user_catalog,
            result,
            next(c for c in result.class_bodies if c.class_name == "NightlyJob"),
        )
        assert dossier.is_excluded_by_anti_pattern
        assert dossier.anti_pattern_matches[0].label == "scheduled_job"

    def test_user_protocol_signature_remote_proxy(self) -> None:
        source = """
        class Caller:
            def go(self, client):
                return client.myorg_remote_agent_invoke(payload={})
        """
        user_catalog = AgentSignatureCatalog(
            protocols=[
                AgentProtocolSignature(
                    id="myorg.remote",
                    protocol="remote_http",
                    evidence_pattern="remote_proxy",
                    role="client",
                    qualified_name_substrings=("myorg_remote_agent_invoke",),
                )
            ]
        )
        result = _parse(source)
        dossier = build_dossier_for_class(
            user_catalog,
            result,
            next(c for c in result.class_bodies if c.class_name == "Caller"),
        )
        assert dossier.has_remote_proxy_evidence


# ---------------------------------------------------------------------------
# Bulk helpers
# ---------------------------------------------------------------------------


class TestBulkBuild:
    def test_build_dossiers_across_multiple_files(self) -> None:
        sources = [
            (
                "svc_a.py",
                """
                from langchain.agents import AgentExecutor

                class ServiceA:
                    def run(self, tools):
                        return AgentExecutor.from_agent_and_tools(tools=tools, agent=None)
                """,
            ),
            (
                "svc_b.py",
                """
                from pydantic import BaseModel

                class Request(BaseModel):
                    text: str
                """,
            ),
        ]
        catalog = default_catalog()
        results = [_parse(src, file_path=path) for path, src in sources]
        dossiers = build_dossiers(catalog, results)
        by_name = {d.class_name: d for d in dossiers}
        assert "ServiceA" in by_name
        assert "Request" in by_name
        assert by_name["ServiceA"].has_direct_agent_evidence
        assert by_name["Request"].is_excluded_by_anti_pattern

    def test_build_dossiers_skips_files_with_no_classes(self) -> None:
        src = """
        import os

        def top_level():
            return os.getpid()
        """
        result = _parse(src)
        dossiers = build_dossiers(default_catalog(), [result])
        assert dossiers == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_base_class_suffix_match(self) -> None:
        """A qualified base such as ``langchain.agents.BaseSingleActionAgent``
        should match a sig declared with the short name.
        """
        source = """
        import langchain.agents

        class X(langchain.agents.BaseSingleActionAgent):
            pass
        """
        dossier = _dossier_for(source, "X")
        ids = {m.signature_id for m in dossier.framework_matches}
        assert "langchain.BaseSingleActionAgent" in ids

    def test_multiple_classes_in_one_file(self) -> None:
        source = """
        from langchain.agents import AgentExecutor
        from pydantic import BaseModel

        class Agent1:
            def run(self, tools):
                return AgentExecutor.from_agent_and_tools(tools=tools, agent=None)

        class SomeData(BaseModel):
            name: str
        """
        result = _parse(source)
        dossiers = build_dossiers(default_catalog(), [result])
        by_name = {d.class_name: d for d in dossiers}
        assert by_name["Agent1"].has_direct_agent_evidence
        assert by_name["SomeData"].is_excluded_by_anti_pattern

    def test_class_without_any_match_has_empty_dossier(self) -> None:
        source = """
        class Plain:
            x = 1
        """
        dossier = _dossier_for(source, "Plain")
        assert dossier.framework_matches == []
        assert dossier.protocol_matches == []
        assert dossier.react_loop_matches == []
        assert dossier.anti_pattern_matches == []
        assert dossier.has_direct_agent_evidence is False
        assert dossier.preferred_pattern is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
