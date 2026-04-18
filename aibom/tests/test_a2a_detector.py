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

"""Tests for the offline A2A Agent Card detector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aibom.models import ScanContext
from aibom.models.enums import AIComponentType, DetectionSource
from aibom.scanners.a2a_detector import (
    A2ADetector,
    _is_well_known_agent_card_path,
    _looks_like_agent_card_shape,
    _normalize_agent_card,
    _redacted_security_summary,
)
from aibom.scanners.file_cache import clear_cache


@pytest.fixture(autouse=True)
def _reset_cache():
    clear_cache()
    yield
    clear_cache()


class TestWellKnownPathMatching:
    @pytest.mark.parametrize(
        "path",
        [
            "/repo/.well-known/agent-card.json",
            "/a/b/c/.well-known/agent.json",
            "services/myagent/.well-known/agent-card.json",
        ],
    )
    def test_matches_canonical_paths(self, path: str) -> None:
        assert _is_well_known_agent_card_path(Path(path))

    @pytest.mark.parametrize(
        "path",
        [
            "/repo/agent.json",
            "/repo/.well-known/other.json",
            "/repo/well-known/agent.json",
            "/repo/.well-known-stuff/agent.json",
            "/repo/.well-known/agent.json.bak",
        ],
    )
    def test_rejects_non_canonical_paths(self, path: str) -> None:
        assert not _is_well_known_agent_card_path(Path(path))


class TestAgentCardShape:
    def test_rejects_non_dict(self) -> None:
        assert _looks_like_agent_card_shape(None) == (False, "")
        assert _looks_like_agent_card_shape([]) == (False, "")
        assert _looks_like_agent_card_shape("name") == (False, "")

    def test_rejects_empty_dict(self) -> None:
        assert _looks_like_agent_card_shape({}) == (False, "")

    def test_rejects_missing_name(self) -> None:
        assert _looks_like_agent_card_shape(
            {"description": "x", "skills": [{"id": "y", "name": "y"}]}
        ) == (False, "")

    def test_rejects_numeric_name(self) -> None:
        assert _looks_like_agent_card_shape({"name": 42}) == (False, "")

    def test_accepts_well_formed_skills(self) -> None:
        ok, reason = _looks_like_agent_card_shape(
            {"name": "x", "skills": [{"id": "foo", "name": "Foo"}]}
        )
        assert ok
        assert reason == "skills_array"

    def test_rejects_skills_without_id_or_name(self) -> None:
        ok, _ = _looks_like_agent_card_shape(
            {"name": "x", "skills": [{"tags": ["t"]}]}
        )
        assert not ok

    def test_accepts_supported_interfaces_camel_case(self) -> None:
        ok, reason = _looks_like_agent_card_shape(
            {
                "name": "x",
                "supportedInterfaces": [
                    {"url": "https://a", "protocolBinding": "JSONRPC"}
                ],
            }
        )
        assert ok
        assert reason == "supported_interfaces"

    def test_accepts_supported_interfaces_snake_case(self) -> None:
        ok, _ = _looks_like_agent_card_shape(
            {
                "name": "x",
                "supported_interfaces": [
                    {"url": "https://a", "protocol_binding": "JSONRPC"}
                ],
            }
        )
        assert ok

    def test_accepts_api_type_a2a(self) -> None:
        ok, reason = _looks_like_agent_card_shape(
            {"name": "x", "api": {"type": "a2a", "url": "https://a"}}
        )
        assert ok
        assert reason == "api_type_a2a"

    def test_accepts_capabilities_dict(self) -> None:
        ok, reason = _looks_like_agent_card_shape(
            {"name": "x", "capabilities": {"streaming": True}}
        )
        assert ok
        assert reason == "capabilities_dict"

    def test_rejects_capabilities_as_string_list(self) -> None:
        ok, _ = _looks_like_agent_card_shape(
            {"name": "x", "capabilities": ["task_execution"]}
        )
        assert not ok

    def test_accepts_legacy_identity_schema(self) -> None:
        ok, reason = _looks_like_agent_card_shape(
            {
                "name": "x",
                "identity": {"name": "ExampleAgent", "version": "1.0"},
                "skills": [{"id": "data", "name": "Data"}],
            }
        )
        assert ok
        assert reason == "skills_array"

    def test_rejects_package_json_shape(self) -> None:
        """``package.json`` looks similar but is not an Agent Card."""
        ok, _ = _looks_like_agent_card_shape(
            {"name": "my-app", "version": "1.0", "dependencies": {"express": "^4"}}
        )
        assert not ok

    def test_rejects_tsconfig_shape(self) -> None:
        ok, _ = _looks_like_agent_card_shape(
            {
                "name": "project",
                "compilerOptions": {"target": "es2020"},
                "include": ["src/**/*"],
            }
        )
        assert not ok


class TestNormalize:
    def test_empty_input(self) -> None:
        assert _normalize_agent_card({}) == {}
        assert _normalize_agent_card(None) == {}  # type: ignore[arg-type]
        assert _normalize_agent_card("x") == {}  # type: ignore[arg-type]

    def test_canonicalizes_camel_case_to_snake(self) -> None:
        out = _normalize_agent_card(
            {
                "name": "A",
                "description": "B",
                "version": "1.0",
                "documentationUrl": "https://docs.example.com",
                "iconUrl": "https://icon.example.com",
                "defaultInputModes": ["text/plain"],
                "defaultOutputModes": ["application/json"],
            }
        )
        assert out["name"] == "A"
        assert out["description"] == "B"
        assert out["version"] == "1.0"
        assert out["documentation_url"] == "https://docs.example.com"
        assert out["icon_url"] == "https://icon.example.com"
        assert out["default_input_modes"] == ["text/plain"]
        assert out["default_output_modes"] == ["application/json"]

    def test_strips_whitespace_around_strings(self) -> None:
        out = _normalize_agent_card({"name": "  A  ", "version": " 1 "})
        assert out["name"] == "A"
        assert out["version"] == "1"

    def test_provider_summary(self) -> None:
        out = _normalize_agent_card(
            {
                "name": "A",
                "provider": {
                    "organization": "Example Inc.",
                    "url": "https://example.com",
                    "internal_secret": "should-not-be-copied",
                },
            }
        )
        assert out["provider"] == {
            "organization": "Example Inc.",
            "url": "https://example.com",
        }
        assert "internal_secret" not in out["provider"]

    def test_capabilities_dict(self) -> None:
        out = _normalize_agent_card(
            {
                "name": "A",
                "capabilities": {
                    "streaming": True,
                    "pushNotifications": False,
                    "stateTransitionHistory": True,
                },
            }
        )
        assert out["capabilities"] == {
            "streaming": True,
            "pushNotifications": False,
            "stateTransitionHistory": True,
        }

    def test_capabilities_list_fallback(self) -> None:
        out = _normalize_agent_card(
            {
                "name": "A",
                "capabilities": ["task_execution", "in_task_auth"],
            }
        )
        assert out["capabilities_list"] == ["task_execution", "in_task_auth"]
        assert "capabilities" not in out

    def test_skills_summary_snake_and_camel(self) -> None:
        out = _normalize_agent_card(
            {
                "name": "A",
                "skills": [
                    {
                        "id": "s1",
                        "name": "Skill One",
                        "description": "d",
                        "tags": ["a", "b"],
                        "examples": ["e1", "e2"],
                        "inputModes": ["text/plain"],
                        "outputModes": ["application/json"],
                    },
                    {
                        "id": "s2",
                        "name": "Skill Two",
                        "input_modes": ["text/plain"],
                    },
                    "not a dict",
                    {"tags": ["only-tags"]},
                ],
            }
        )
        assert len(out["skills"]) == 2
        assert out["skills"][0]["id"] == "s1"
        assert out["skills"][0]["input_modes"] == ["text/plain"]
        assert out["skills"][1]["id"] == "s2"

    def test_supported_interfaces_summary(self) -> None:
        out = _normalize_agent_card(
            {
                "name": "A",
                "supportedInterfaces": [
                    {
                        "url": "https://a/v1",
                        "protocolBinding": "JSONRPC",
                        "protocolVersion": "1.0",
                    },
                    {
                        "url": "https://a/grpc",
                        "protocol_binding": "GRPC",
                    },
                    "not a dict",
                ],
            }
        )
        assert len(out["supported_interfaces"]) == 2
        assert out["supported_interfaces"][0]["url"] == "https://a/v1"
        assert out["supported_interfaces"][0]["protocol_binding"] == "JSONRPC"
        assert out["supported_interfaces"][1]["protocol_binding"] == "GRPC"

    def test_api_summary(self) -> None:
        out = _normalize_agent_card(
            {
                "name": "A",
                "api": {"type": "a2a", "url": "https://a/v1", "protocolVersion": "1.0"},
            }
        )
        assert out["api"]["type"] == "a2a"
        assert out["api"]["url"] == "https://a/v1"
        assert out["api"]["protocol_version"] == "1.0"

    def test_signatures_redacted(self) -> None:
        out = _normalize_agent_card(
            {
                "name": "A",
                "signatures": [
                    {
                        "protected": "eyJh...SHOULD-NOT-APPEAR...",
                        "signature": "abcdef...SHOULD-NOT-APPEAR",
                    }
                ],
            }
        )
        assert out["signatures_present"] is True
        assert out["signatures_count"] == 1
        assert "SHOULD-NOT-APPEAR" not in json.dumps(out)

    def test_endpoints_flattened_and_deduped(self) -> None:
        out = _normalize_agent_card(
            {
                "name": "A",
                "supportedInterfaces": [
                    {"url": "https://a/v1"},
                    {"url": "https://a/grpc"},
                    {"url": "https://a/v1"},
                ],
                "api": {"type": "a2a", "url": "https://a/v1"},
            }
        )
        assert out["endpoints"] == ["https://a/v1", "https://a/grpc"]

    def test_security_schemes_types_only_no_urls(self) -> None:
        out = _normalize_agent_card(
            {
                "name": "A",
                "securitySchemes": {
                    "google_sso": {
                        "openIdConnectSecurityScheme": {
                            "openIdConnectUrl": (
                                "https://accounts.google.com/.well-known/openid-config"
                            )
                        }
                    },
                    "api_key_v1": {"type": "apiKey"},
                },
            }
        )
        assert "accounts.google.com" not in json.dumps(out)
        summary = out["security_schemes"]
        assert "openIdConnect" in summary["scheme_ids_by_type"]
        assert summary["scheme_ids_by_type"]["openIdConnect"] == ["google_sso"]
        assert summary["scheme_ids_by_type"]["apiKey"] == ["api_key_v1"]


class TestRedactedSecuritySummary:
    def test_empty_input(self) -> None:
        assert _redacted_security_summary({}) == {}
        assert _redacted_security_summary(None) == {}  # type: ignore[arg-type]

    def test_handles_unknown_scheme_shape(self) -> None:
        out = _redacted_security_summary(
            {
                "weirdThing": {"something": "else"},
            }
        )
        assert out["scheme_ids_by_type"]["unknown"] == ["weirdThing"]

    def test_sorts_ids_deterministically(self) -> None:
        out = _redacted_security_summary(
            {
                "z_key": {"type": "apiKey"},
                "a_key": {"type": "apiKey"},
                "m_key": {"type": "apiKey"},
            }
        )
        assert out["scheme_ids_by_type"]["apiKey"] == ["a_key", "m_key", "z_key"]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


class TestScanJsonFiles:
    def test_well_known_agent_card_emits_agent(self, tmp_path: Path) -> None:
        _write(
            tmp_path / ".well-known" / "agent-card.json",
            json.dumps(
                {
                    "name": "WeatherAgent",
                    "description": "Weather forecasts",
                    "version": "2.0.0",
                    "supportedInterfaces": [
                        {
                            "url": "https://w.example.com/a2a",
                            "protocolBinding": "JSONRPC",
                        }
                    ],
                    "skills": [{"id": "forecast", "name": "Forecast"}],
                }
            ),
        )
        det = A2ADetector()
        comps, rels = det.scan(ScanContext(paths=[str(tmp_path)]))
        assert rels == []
        assert len(comps) == 1
        c = comps[0]
        assert c.component_type == AIComponentType.AGENT
        assert c.framework == "a2a"
        assert c.detection_source == DetectionSource.CONFIG_FILE
        assert c.name == "WeatherAgent"
        assert c.metadata["source_shape"] == "json"
        assert c.metadata["well_known"] is True
        card = c.metadata["agent_card"]
        assert card["version"] == "2.0.0"
        assert card["endpoints"] == ["https://w.example.com/a2a"]
        assert [s["id"] for s in card["skills"]] == ["forecast"]

    def test_legacy_well_known_agent_json_filename(self, tmp_path: Path) -> None:
        _write(
            tmp_path / ".well-known" / "agent.json",
            json.dumps(
                {
                    "name": "LegacyAgent",
                    "description": "d",
                    "version": "1",
                    "skills": [{"id": "x", "name": "X"}],
                }
            ),
        )
        det = A2ADetector()
        comps, _ = det.scan(ScanContext(paths=[str(tmp_path)]))
        assert len(comps) == 1
        assert comps[0].metadata["well_known"] is True

    def test_arbitrary_path_with_strong_shape_still_detected(
        self, tmp_path: Path
    ) -> None:
        _write(
            tmp_path / "docs" / "my-agent-card.json",
            json.dumps(
                {
                    "name": "DocsAgent",
                    "description": "d",
                    "skills": [{"id": "x", "name": "X"}],
                }
            ),
        )
        det = A2ADetector()
        comps, _ = det.scan(ScanContext(paths=[str(tmp_path)]))
        assert len(comps) == 1
        assert comps[0].metadata["well_known"] is False
        assert comps[0].metadata["shape_reason"] == "skills_array"

    def test_package_json_ignored(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "package.json",
            json.dumps({"name": "my-app", "version": "1.0.0", "dependencies": {}}),
        )
        det = A2ADetector()
        comps, _ = det.scan(ScanContext(paths=[str(tmp_path)]))
        assert comps == []

    def test_tsconfig_ignored(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "tsconfig.json",
            json.dumps({"compilerOptions": {"target": "es2020"}}),
        )
        det = A2ADetector()
        comps, _ = det.scan(ScanContext(paths=[str(tmp_path)]))
        assert comps == []

    def test_malformed_json_ignored(self, tmp_path: Path) -> None:
        _write(tmp_path / "broken.json", "{invalid json")
        det = A2ADetector()
        comps, _ = det.scan(ScanContext(paths=[str(tmp_path)]))
        assert comps == []

    def test_json_array_ignored(self, tmp_path: Path) -> None:
        """Agent Cards must be objects; a bare JSON array must not match."""
        _write(
            tmp_path / "list.json",
            json.dumps([{"name": "x", "skills": [{"id": "s"}]}]),
        )
        det = A2ADetector()
        comps, _ = det.scan(ScanContext(paths=[str(tmp_path)]))
        assert comps == []

    def test_well_known_with_unknown_shape_still_detected(
        self, tmp_path: Path
    ) -> None:
        """A file literally named ``.well-known/agent-card.json`` is
        trusted even if shape-checking is inconclusive — path wins."""
        _write(
            tmp_path / ".well-known" / "agent-card.json",
            json.dumps({"name": "WeirdShape"}),
        )
        det = A2ADetector()
        comps, _ = det.scan(ScanContext(paths=[str(tmp_path)]))
        assert len(comps) == 1
        assert comps[0].name == "WeirdShape"
        assert comps[0].metadata["shape_reason"] == "well_known_path"

    def test_oversized_json_skipped(self, tmp_path: Path, monkeypatch) -> None:
        """Files bigger than ``_MAX_JSON_SIZE_BYTES`` are not parsed."""
        from aibom.scanners import a2a_detector as mod

        monkeypatch.setattr(mod, "_MAX_JSON_SIZE_BYTES", 16)
        _write(
            tmp_path / ".well-known" / "agent-card.json",
            json.dumps(
                {"name": "Big", "description": "d", "skills": [{"id": "a", "name": "b"}]}
            ),
        )
        det = A2ADetector()
        comps, _ = det.scan(ScanContext(paths=[str(tmp_path)]))
        assert comps == []


class TestScanPythonFiles:
    def test_a2a_server_with_variable_resolution(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "server.py",
            """
from a2a.server import A2AServer, AgentCard

my_card = AgentCard(
    name='SupportAgent',
    description='Customer support agent',
    version='1.0.0',
    skills=[{'id': 'refund', 'name': 'Process Refund'}],
)

app = A2AServer(agent_card=my_card)
""",
        )
        det = A2ADetector()
        comps, _ = det.scan(ScanContext(paths=[str(tmp_path)]))
        assert len(comps) == 1
        c = comps[0]
        assert c.name == "SupportAgent"
        assert c.framework == "a2a"
        assert c.detection_source == DetectionSource.CODE_ANALYSIS
        assert c.metadata["source_shape"] == "python"
        assert c.metadata["constructor"] == "A2AServer"
        card = c.metadata["agent_card"]
        assert card["version"] == "1.0.0"
        assert [s["id"] for s in card["skills"]] == ["refund"]

    def test_a2a_server_with_inline_dict(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "inline.py",
            """
from a2a.server import A2AServer

app = A2AServer(
    agent_card={
        'name': 'InlineAgent',
        'description': 'd',
        'version': '0.1',
        'skills': [{'id': 'do_thing', 'name': 'Do Thing'}],
    }
)
""",
        )
        det = A2ADetector()
        comps, _ = det.scan(ScanContext(paths=[str(tmp_path)]))
        assert len(comps) == 1
        c = comps[0]
        assert c.name == "InlineAgent"
        assert c.metadata["constructor"] == "A2AServer"
        assert [s["id"] for s in c.metadata["agent_card"]["skills"]] == [
            "do_thing"
        ]

    def test_bare_agent_card_declaration_emits(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "card_only.py",
            """
from a2a.types import AgentCard

bare = AgentCard(
    name='BareAgent',
    description='d',
    version='1',
    skills=[{'id': 'x', 'name': 'X'}],
)
""",
        )
        det = A2ADetector()
        comps, _ = det.scan(ScanContext(paths=[str(tmp_path)]))
        assert len(comps) == 1
        assert comps[0].name == "BareAgent"
        assert comps[0].metadata["constructor"] == "AgentCard"

    def test_server_and_its_resolved_card_do_not_double_emit(
        self, tmp_path: Path
    ) -> None:
        _write(
            tmp_path / "server.py",
            """
from a2a.server import A2AServer, AgentCard

card = AgentCard(name='X', description='d', version='1', skills=[{'id': 'a', 'name': 'A'}])
app = A2AServer(agent_card=card)
""",
        )
        det = A2ADetector()
        comps, _ = det.scan(ScanContext(paths=[str(tmp_path)]))
        constructors = sorted(c.metadata["constructor"] for c in comps)
        assert constructors == ["A2AServer"]

    def test_multiple_servers_in_one_file(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "many.py",
            """
from a2a.server import A2AServer, AgentCard

a = AgentCard(name='A', description='d', version='1', skills=[{'id': 'x', 'name': 'X'}])
b = AgentCard(name='B', description='d', version='1', skills=[{'id': 'y', 'name': 'Y'}])

app_a = A2AServer(agent_card=a)
app_b = A2AServer(agent_card=b)
""",
        )
        det = A2ADetector()
        comps, _ = det.scan(ScanContext(paths=[str(tmp_path)]))
        names = sorted(c.name for c in comps)
        assert names == ["A", "B"]
        assert all(c.metadata["constructor"] == "A2AServer" for c in comps)

    def test_call_with_unresolvable_variable_kwargs_drops_junk(
        self, tmp_path: Path
    ) -> None:
        """VARIABLE: / ATTRIBUTE: references must not leak into the card."""
        _write(
            tmp_path / "var.py",
            """
from a2a.types import AgentCard

shared = load_shared()

card = AgentCard(
    name='V',
    description='d',
    version='1',
    skills=[{'id': 'a', 'name': 'A'}],
    metadata=shared,
)
""",
        )
        det = A2ADetector()
        comps, _ = det.scan(ScanContext(paths=[str(tmp_path)]))
        assert len(comps) == 1
        card = comps[0].metadata["agent_card"]
        assert "metadata" not in json.dumps(card)
        assert card["name"] == "V"

    def test_unrelated_python_file_skipped_cheaply(self, tmp_path: Path) -> None:
        _write(tmp_path / "u.py", "def foo(): return 42\nclass Bar: pass\n")
        det = A2ADetector()
        comps, _ = det.scan(ScanContext(paths=[str(tmp_path)]))
        assert comps == []

    def test_syntax_error_does_not_crash(self, tmp_path: Path) -> None:
        _write(tmp_path / "broken.py", "from a2a import AgentCard\ndef bad(:\n")
        det = A2ADetector()
        comps, _ = det.scan(ScanContext(paths=[str(tmp_path)]))
        assert comps == []

    def test_a2a_server_without_payload_is_not_emitted(
        self, tmp_path: Path
    ) -> None:
        """An ``A2AServer()`` call with no resolvable Agent Card payload
        yields no component — the detector needs at least some identity
        or structural field to emit anything meaningful."""
        _write(
            tmp_path / "empty.py",
            """
from a2a.server import A2AServer

cfg = load_from_config()
app = A2AServer(agent_card=cfg)
""",
        )
        det = A2ADetector()
        comps, _ = det.scan(ScanContext(paths=[str(tmp_path)]))
        assert comps == []


class TestA2ADetectorIntegration:
    def test_scanner_is_registered_auto_discovery(self) -> None:
        """Importing the scanner registers it in the global registry."""
        from aibom.scanners import a2a_detector  # noqa: F401
        from aibom.scanners.base import scanner_registry

        assert A2ADetector in scanner_registry

    def test_detector_supports_any_context(self, tmp_path: Path) -> None:
        det = A2ADetector()
        assert det.supports(ScanContext(paths=[str(tmp_path)]))

    def test_empty_scan_yields_no_components(self, tmp_path: Path) -> None:
        det = A2ADetector()
        comps, rels = det.scan(ScanContext(paths=[str(tmp_path)]))
        assert comps == []
        assert rels == []

    def test_exclude_patterns_honored(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "vendor" / ".well-known" / "agent-card.json",
            json.dumps(
                {
                    "name": "Vendored",
                    "description": "d",
                    "skills": [{"id": "x", "name": "X"}],
                }
            ),
        )
        _write(
            tmp_path / ".well-known" / "agent-card.json",
            json.dumps(
                {
                    "name": "OwnAgent",
                    "description": "d",
                    "skills": [{"id": "x", "name": "X"}],
                }
            ),
        )
        det = A2ADetector()
        comps, _ = det.scan(
            ScanContext(paths=[str(tmp_path)], exclude_patterns=["vendor/**"])
        )
        names = [c.name for c in comps]
        assert "OwnAgent" in names
        assert "Vendored" not in names

    def test_empty_paths_list(self) -> None:
        det = A2ADetector()
        comps, rels = det.scan(ScanContext(paths=[]))
        assert comps == []
        assert rels == []
