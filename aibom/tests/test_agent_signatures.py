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

"""Tests for the agent-signature catalog (dataclasses, defaults, YAML
override parser, and merge logic)."""

from __future__ import annotations

import logging

from aibom.agent_signatures import (
    AgentAntiPatternSignature,
    AgentFrameworkSignature,
    AgentProtocolSignature,
    AgentSignatureCatalog,
    VerificationPolicy,
    default_catalog,
    merge_catalogs,
    parse_user_signatures,
    resolve_catalog,
)


class TestDefaultCatalog:
    def test_default_catalog_is_populated(self) -> None:
        cat = default_catalog()
        assert len(cat.frameworks) >= 5
        assert len(cat.protocols) >= 4
        assert len(cat.anti_patterns) >= 4

    def test_default_catalog_returns_independent_copies(self) -> None:
        """Mutating one copy must not affect a subsequent call."""
        cat_a = default_catalog()
        cat_a.frameworks.clear()
        cat_a.protocols.clear()
        cat_a.anti_patterns.clear()

        cat_b = default_catalog()
        assert cat_b.frameworks, "default_catalog must return a fresh copy"
        assert cat_b.protocols
        assert cat_b.anti_patterns

    def test_default_catalog_includes_known_frameworks(self) -> None:
        cat = default_catalog()
        ids = {s.id for s in cat.frameworks}
        assert "langchain.AgentExecutor" in ids
        assert "langgraph.create_react_agent" in ids
        assert "autogen.AssistantAgent" in ids
        assert "crewai.Agent" in ids
        assert "llama_index.BaseAgent" in ids
        assert "strands.Agent" in ids
        assert "strands.experimental.BidiAgent" in ids

    def test_strands_signatures_require_strands_import(self) -> None:
        """Short name ``Agent`` is too ambiguous without the ``strands``
        import substring to disambiguate against CrewAI/OpenAI Agents/etc."""
        cat = default_catalog()
        strands_sigs = [s for s in cat.frameworks if s.framework == "strands"]
        assert len(strands_sigs) == 2, (
            f"expected exactly two strands signatures, got {len(strands_sigs)}"
        )
        for sig in strands_sigs:
            assert any("strands" in sub for sub in sig.import_substrings), (
                f"{sig.id} must constrain matching via a strands import substring"
            )

    def test_default_catalog_includes_a2a_and_mcp_protocols(self) -> None:
        cat = default_catalog()
        ids = {s.id for s in cat.protocols}
        assert "a2a.server" in ids
        assert "a2a.client" in ids
        assert "mcp.server" in ids
        assert "mcp.client" in ids
        assert "openai.assistants" in ids

    def test_mcp_protocols_have_non_agent_pattern(self) -> None:
        """MCP usage alone is NOT evidence of agency."""
        cat = default_catalog()
        mcp_sigs = [s for s in cat.protocols if s.protocol == "mcp"]
        assert mcp_sigs, "MCP signatures must be present in defaults"
        for sig in mcp_sigs:
            assert sig.evidence_pattern == "other", (
                f"MCP sig '{sig.id}' must use pattern 'other' so it does not "
                "imply agency on its own"
            )

    def test_a2a_server_is_positive_agent_evidence(self) -> None:
        cat = default_catalog()
        a2a_server = next(s for s in cat.protocols if s.id == "a2a.server")
        assert a2a_server.evidence_pattern == "a2a_server"

    def test_a2a_client_is_remote_proxy(self) -> None:
        cat = default_catalog()
        a2a_client = next(s for s in cat.protocols if s.id == "a2a.client")
        assert a2a_client.evidence_pattern == "remote_proxy"

    def test_default_catalog_includes_known_anti_patterns(self) -> None:
        cat = default_catalog()
        labels = {s.label for s in cat.anti_patterns}
        assert "temporal_workflow" in labels
        assert "celery_task" in labels
        assert "airflow_dag" in labels
        assert "fastapi_endpoint" in labels
        assert "pydantic_basemodel" in labels

    def test_default_verification_policy(self) -> None:
        cat = default_catalog()
        assert cat.verification_policy.require_evidence_for_agent is True
        assert cat.verification_policy.allow_remote_proxy_without_cross_repo is False
        assert cat.verification_policy.min_react_loop_call_count >= 2
        assert cat.verification_policy.min_react_loop_distinct_callees >= 2

    def test_every_framework_sig_has_non_empty_matcher(self) -> None:
        """A framework sig must declare at least one matching surface."""
        for sig in default_catalog().frameworks:
            assert (
                sig.entrypoint_qualified_names
                or sig.base_class_names
            ), f"framework sig '{sig.id}' has no matching surface"

    def test_every_protocol_sig_has_non_empty_matcher(self) -> None:
        for sig in default_catalog().protocols:
            assert (
                sig.import_substrings
                or sig.qualified_name_substrings
                or sig.string_literal_substrings
            ), f"protocol sig '{sig.id}' has no matching surface"


class TestIsEmpty:
    def test_fresh_catalog_is_empty(self) -> None:
        assert AgentSignatureCatalog().is_empty

    def test_catalog_with_framework_is_not_empty(self) -> None:
        cat = AgentSignatureCatalog(
            frameworks=[
                AgentFrameworkSignature(
                    id="x",
                    framework="y",
                    evidence_pattern="framework_agent",
                )
            ]
        )
        assert not cat.is_empty

    def test_catalog_with_custom_policy_is_not_empty(self) -> None:
        cat = AgentSignatureCatalog(
            verification_policy=VerificationPolicy(require_evidence_for_agent=False)
        )
        assert not cat.is_empty


class TestParseUserSignatures:
    def test_empty_input_returns_empty_catalog(self) -> None:
        assert parse_user_signatures({}).is_empty
        assert parse_user_signatures(None).is_empty

    def test_non_dict_input_returns_empty_catalog(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="aibom.agent_signatures"):
            result = parse_user_signatures(["not", "a", "dict"])
        assert result.is_empty
        assert any(
            "must be a mapping" in rec.message for rec in caplog.records
        )

    def test_parse_framework(self) -> None:
        raw = {
            "frameworks": [
                {
                    "id": "myorg.InternalAgent",
                    "framework": "myorg",
                    "evidence_pattern": "framework_inheritance",
                    "base_class_names": ["InternalAgentBase"],
                    "import_substrings": "myorg.agents",
                    "description": "internal",
                }
            ]
        }
        result = parse_user_signatures(raw)
        assert len(result.frameworks) == 1
        sig = result.frameworks[0]
        assert sig.id == "myorg.InternalAgent"
        assert sig.framework == "myorg"
        assert sig.evidence_pattern == "framework_inheritance"
        assert sig.base_class_names == ("InternalAgentBase",)
        assert sig.import_substrings == ("myorg.agents",)

    def test_parse_framework_missing_id_is_skipped(self, caplog) -> None:
        raw = {
            "frameworks": [
                {"framework": "x", "evidence_pattern": "framework_agent"},
                {
                    "id": "keep.me",
                    "evidence_pattern": "framework_agent",
                },
            ]
        }
        with caplog.at_level(logging.WARNING, logger="aibom.agent_signatures"):
            result = parse_user_signatures(raw)
        assert len(result.frameworks) == 1
        assert result.frameworks[0].id == "keep.me"
        assert any("'id' is required" in rec.message for rec in caplog.records)

    def test_parse_framework_missing_pattern_is_skipped(self, caplog) -> None:
        raw = {"frameworks": [{"id": "a", "framework": "b"}]}
        with caplog.at_level(logging.WARNING, logger="aibom.agent_signatures"):
            result = parse_user_signatures(raw)
        assert result.frameworks == []
        assert any(
            "'evidence_pattern' is required" in rec.message for rec in caplog.records
        )

    def test_parse_protocol(self) -> None:
        raw = {
            "protocols": [
                {
                    "id": "myorg.remote",
                    "protocol": "remote_http",
                    "evidence_pattern": "remote_proxy",
                    "role": "client",
                    "qualified_name_substrings": ["myorg.remote.Client"],
                    "string_literal_substrings": ["/api/v1/agent"],
                }
            ]
        }
        result = parse_user_signatures(raw)
        assert len(result.protocols) == 1
        sig = result.protocols[0]
        assert sig.id == "myorg.remote"
        assert sig.protocol == "remote_http"
        assert sig.evidence_pattern == "remote_proxy"
        assert sig.role == "client"
        assert sig.qualified_name_substrings == ("myorg.remote.Client",)
        assert sig.string_literal_substrings == ("/api/v1/agent",)

    def test_parse_protocol_missing_fields_skipped(self) -> None:
        raw = {
            "protocols": [
                {"id": "x"},  # missing protocol and pattern
                {"id": "y", "protocol": "p"},  # missing pattern
                {"id": "z", "protocol": "p", "evidence_pattern": "other"},  # OK
            ]
        }
        result = parse_user_signatures(raw)
        assert len(result.protocols) == 1
        assert result.protocols[0].id == "z"

    def test_parse_anti_pattern(self) -> None:
        raw = {
            "anti_patterns": [
                {
                    "id": "myorg.scheduled_job",
                    "label": "scheduled_job",
                    "base_class_names": ["ScheduledJob"],
                    "import_substrings": ["myorg.jobs"],
                    "description": "scheduled",
                }
            ]
        }
        result = parse_user_signatures(raw)
        assert len(result.anti_patterns) == 1
        sig = result.anti_patterns[0]
        assert sig.id == "myorg.scheduled_job"
        assert sig.label == "scheduled_job"
        assert sig.base_class_names == ("ScheduledJob",)

    def test_parse_anti_pattern_missing_label_skipped(self) -> None:
        raw = {"anti_patterns": [{"id": "a"}]}
        result = parse_user_signatures(raw)
        assert result.anti_patterns == []

    def test_parse_verification_policy(self) -> None:
        raw = {
            "verification_policy": {
                "require_evidence_for_agent": False,
                "allow_remote_proxy_without_cross_repo": True,
                "min_react_loop_call_count": 3,
                "min_react_loop_distinct_callees": 4,
            }
        }
        result = parse_user_signatures(raw)
        policy = result.verification_policy
        assert policy.require_evidence_for_agent is False
        assert policy.allow_remote_proxy_without_cross_repo is True
        assert policy.min_react_loop_call_count == 3
        assert policy.min_react_loop_distinct_callees == 4

    def test_parse_verification_policy_partial_override(self) -> None:
        """Fields not supplied fall back to built-in defaults."""
        raw = {
            "verification_policy": {
                "min_react_loop_call_count": 5,
            }
        }
        result = parse_user_signatures(raw)
        policy = result.verification_policy
        assert policy.min_react_loop_call_count == 5
        assert policy.require_evidence_for_agent is True
        assert policy.allow_remote_proxy_without_cross_repo is False

    def test_parse_unknown_keys_are_warned(self, caplog) -> None:
        raw = {"frameworks": [], "mystery_key": []}
        with caplog.at_level(logging.WARNING, logger="aibom.agent_signatures"):
            parse_user_signatures(raw)
        assert any(
            "'agent_signatures.mystery_key'" in rec.message for rec in caplog.records
        )

    def test_as_tuple_of_str_handles_scalar_and_list(self) -> None:
        """Scalars should be coerced to a one-element tuple; non-str values
        should be stringified.
        """
        raw = {
            "frameworks": [
                {
                    "id": "a",
                    "evidence_pattern": "framework_agent",
                    "entrypoint_qualified_names": "one.name",
                    "base_class_names": ["B1", "B2"],
                }
            ]
        }
        result = parse_user_signatures(raw)
        sig = result.frameworks[0]
        assert sig.entrypoint_qualified_names == ("one.name",)
        assert sig.base_class_names == ("B1", "B2")


class TestMergeCatalogs:
    def test_merge_with_none_user_keeps_builtins(self) -> None:
        builtin = default_catalog()
        merged = merge_catalogs(builtin, None)
        assert len(merged.frameworks) == len(builtin.frameworks)
        assert len(merged.protocols) == len(builtin.protocols)
        assert len(merged.anti_patterns) == len(builtin.anti_patterns)

    def test_merge_with_empty_user_keeps_builtins(self) -> None:
        builtin = default_catalog()
        merged = merge_catalogs(builtin, AgentSignatureCatalog())
        assert [s.id for s in merged.frameworks] == [s.id for s in builtin.frameworks]

    def test_merge_appends_new_user_frameworks(self) -> None:
        builtin = default_catalog()
        user = AgentSignatureCatalog(
            frameworks=[
                AgentFrameworkSignature(
                    id="myorg.NewAgent",
                    framework="myorg",
                    evidence_pattern="framework_agent",
                    entrypoint_qualified_names=("myorg.NewAgent",),
                )
            ]
        )
        merged = merge_catalogs(builtin, user)
        assert len(merged.frameworks) == len(builtin.frameworks) + 1
        assert any(s.id == "myorg.NewAgent" for s in merged.frameworks)

    def test_user_entry_overrides_builtin_by_id(self) -> None:
        builtin = default_catalog()
        overridden = AgentFrameworkSignature(
            id="langchain.AgentExecutor",
            framework="langchain-custom",
            evidence_pattern="framework_agent",
            entrypoint_qualified_names=("totally.different.path",),
        )
        user = AgentSignatureCatalog(frameworks=[overridden])
        merged = merge_catalogs(builtin, user)
        sig = next(s for s in merged.frameworks if s.id == "langchain.AgentExecutor")
        assert sig.framework == "langchain-custom"
        assert sig.entrypoint_qualified_names == ("totally.different.path",)

    def test_merge_preserves_order_of_builtins(self) -> None:
        builtin = default_catalog()
        builtin_ids = [s.id for s in builtin.frameworks]
        user = AgentSignatureCatalog(
            frameworks=[
                AgentFrameworkSignature(
                    id="myorg.AgentX",
                    framework="myorg",
                    evidence_pattern="framework_agent",
                )
            ]
        )
        merged = merge_catalogs(builtin, user)
        merged_ids = [s.id for s in merged.frameworks]
        # Built-ins keep their original relative order
        assert merged_ids[: len(builtin_ids)] == builtin_ids
        assert merged_ids[-1] == "myorg.AgentX"

    def test_merge_anti_pattern_override(self) -> None:
        builtin = default_catalog()
        override = AgentAntiPatternSignature(
            id="temporal.workflow",
            label="temporal_workflow_custom",
            decorator_qualified_names=("my.alias.defn",),
            import_substrings=("temporalio.workflow",),
        )
        user = AgentSignatureCatalog(anti_patterns=[override])
        merged = merge_catalogs(builtin, user)
        sig = next(s for s in merged.anti_patterns if s.id == "temporal.workflow")
        assert sig.label == "temporal_workflow_custom"

    def test_merge_protocol_new_entry(self) -> None:
        builtin = default_catalog()
        new_proto = AgentProtocolSignature(
            id="myorg.custom_rpc",
            protocol="custom_rpc",
            evidence_pattern="other",
            role="client",
            qualified_name_substrings=("myorg.rpc.Client",),
        )
        user = AgentSignatureCatalog(protocols=[new_proto])
        merged = merge_catalogs(builtin, user)
        assert any(s.id == "myorg.custom_rpc" for s in merged.protocols)

    def test_merge_with_custom_policy(self) -> None:
        builtin = default_catalog()
        custom_policy = VerificationPolicy(
            require_evidence_for_agent=False, min_react_loop_call_count=7
        )
        user = AgentSignatureCatalog(verification_policy=custom_policy)
        merged = merge_catalogs(builtin, user)
        assert merged.verification_policy == custom_policy


class TestResolveCatalog:
    def test_resolve_with_no_user_returns_defaults(self) -> None:
        cat = resolve_catalog(None)
        ref = default_catalog()
        assert [s.id for s in cat.frameworks] == [s.id for s in ref.frameworks]

    def test_resolve_with_user_applies_overrides(self) -> None:
        user_catalog = parse_user_signatures(
            {
                "frameworks": [
                    {
                        "id": "myorg.Local",
                        "framework": "myorg",
                        "evidence_pattern": "framework_agent",
                        "entrypoint_qualified_names": ["myorg.Local"],
                    }
                ]
            }
        )
        cat = resolve_catalog(user_catalog)
        assert any(s.id == "myorg.Local" for s in cat.frameworks)
