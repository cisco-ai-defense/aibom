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

"""Tests for the A2A cross-repo linker in cross_repo_links.py."""

from __future__ import annotations

from typing import Any

import pytest

from aibom.cross_repo_links import (
    _build_a2a_agent_cross_links,
    build_deterministic_cross_repo_links,
)
from aibom.models import AIComponent
from aibom.models.enums import (
    AIComponentType,
    CrossRepoLinkType,
    DetectionSource,
)


def _make_card(
    name: str,
    endpoints: list[str],
    file_path: str = "/repo-server/.well-known/agent.json",
) -> AIComponent:
    return AIComponent(
        name=name,
        component_type=AIComponentType.AGENT,
        file_path=file_path,
        line_number=0,
        framework="a2a",
        detection_source=DetectionSource.CODE_ANALYSIS,
        metadata={
            "agent_card": {
                "name": name,
                "endpoints": endpoints,
            }
        },
    )


def _make_proxy(
    *,
    name: str,
    remote_url: str,
    status: str = "unverified_url_pattern",
    confidence: float = 0.5,
    file_path: str = "/repo-client/client.py",
    line_number: int = 1,
    matched_instance_id: str = "",
    matched_name: str = "",
    detection_reason: str = "client_constructor_call",
) -> AIComponent:
    return AIComponent(
        name=name,
        component_type=AIComponentType.AGENT_PROXY,
        file_path=file_path,
        line_number=line_number,
        framework="a2a",
        detection_source=DetectionSource.CODE_ANALYSIS,
        heuristic_confidence=confidence,
        needs_agentic=False,
        metadata={
            "remote_url": remote_url,
            "detection_reason": detection_reason,
            "remote_verification": {
                "status": status,
                "confidence": confidence,
                "matched_component_instance_id": matched_instance_id,
                "matched_component_name": matched_name,
                "match_source": "local_scan" if matched_instance_id else "",
            },
        },
    )


def _per_repo(
    client_components: list[AIComponent] = None,
    server_components: list[AIComponent] = None,
) -> dict[str, dict[str, Any]]:
    return {
        "/repo-client": {
            "components": list(client_components or []),
            "relationships": [],
        },
        "/repo-server": {
            "components": list(server_components or []),
            "relationships": [],
        },
    }


class TestBuildA2AAgentCrossLinks:
    def test_verified_proxy_not_cross_linked(self) -> None:
        """Proxies already matched locally are left alone.

        Cross-repo linking is only for proxies whose URL couldn't be
        resolved in the same repo. A verified local match is
        authoritative.
        """
        proxy = _make_proxy(
            name="weather_a2a_proxy",
            remote_url="https://weather.test/a2a",
            status="verified_local_card",
            confidence=1.0,
            matched_instance_id="local-card-id",
            matched_name="WeatherAgent",
        )
        card = _make_card("WeatherAgent", ["https://weather.test/a2a"])
        per_repo = _per_repo(
            client_components=[proxy], server_components=[card]
        )
        links = _build_a2a_agent_cross_links(per_repo)
        assert links == []

    def test_unverified_proxy_matched_across_repos(self) -> None:
        proxy = _make_proxy(
            name="weather_a2a_proxy",
            remote_url="https://weather.test/a2a",
        )
        card = _make_card("WeatherAgent", ["https://weather.test/a2a"])
        per_repo = _per_repo(
            client_components=[proxy], server_components=[card]
        )
        links = _build_a2a_agent_cross_links(per_repo)
        assert len(links) == 1
        link = links[0]
        assert link.link_type == CrossRepoLinkType.A2A_AGENT_CLIENT_SERVER
        assert link.identifier == "https://weather.test/a2a"
        assert link.resolved_value == "WeatherAgent"
        assert len(link.occurrences) == 2
        source_occ = next(o for o in link.occurrences if o.role == "source")
        target_occ = next(o for o in link.occurrences if o.role == "target")
        assert source_occ.repo_path == "/repo-client"
        assert source_occ.component_name == "weather_a2a_proxy"
        assert target_occ.repo_path == "/repo-server"
        assert target_occ.component_name == "WeatherAgent"

    def test_proxy_metadata_upgraded_to_cross_repo_verified(self) -> None:
        """On a cross-repo match, proxy verification is upgraded in place."""
        proxy = _make_proxy(
            name="weather_a2a_proxy",
            remote_url="https://weather.test/a2a",
        )
        card = _make_card("WeatherAgent", ["https://weather.test/a2a"])
        per_repo = _per_repo(
            client_components=[proxy], server_components=[card]
        )
        _build_a2a_agent_cross_links(per_repo)
        updated = per_repo["/repo-client"]["components"][0]
        verification = updated.metadata["remote_verification"]
        assert verification["status"] == "verified_cross_repo_card"
        assert verification["confidence"] == 0.9
        assert verification["match_source"] == "cross_repo"
        assert verification["matched_component_name"] == "WeatherAgent"
        assert verification["matched_repo_path"] == "/repo-server"
        assert updated.heuristic_confidence == 0.9

    def test_heuristic_confidence_not_downgraded(self) -> None:
        """Cross-repo match never lowers an already-high confidence."""
        proxy = _make_proxy(
            name="weather_a2a_proxy",
            remote_url="https://weather.test/a2a",
            confidence=0.95,
        )
        card = _make_card("WeatherAgent", ["https://weather.test/a2a"])
        per_repo = _per_repo(
            client_components=[proxy], server_components=[card]
        )
        _build_a2a_agent_cross_links(per_repo)
        assert per_repo["/repo-client"]["components"][0].heuristic_confidence == 0.95

    def test_same_repo_match_skipped(self) -> None:
        """Proxy + card in the same repo don't produce a cross-repo link.

        If both live in one repo, the local resolver should have
        matched them already; if it didn't we should not paper over
        the miss with a (misleading) cross-repo label.
        """
        proxy = _make_proxy(
            name="weather_a2a_proxy",
            remote_url="https://weather.test/a2a",
        )
        card = _make_card("WeatherAgent", ["https://weather.test/a2a"])
        per_repo = {
            "/repo-same": {
                "components": [proxy, card],
                "relationships": [],
            },
            "/repo-other": {
                "components": [],
                "relationships": [],
            },
        }
        links = _build_a2a_agent_cross_links(per_repo)
        assert links == []

    def test_host_root_match_between_repos(self) -> None:
        """Well-known URL in client should match ``/a2a`` card in server."""
        proxy = _make_proxy(
            name="weather_a2a_proxy",
            remote_url="https://weather.test/.well-known/agent.json",
        )
        card = _make_card("WeatherAgent", ["https://weather.test/a2a"])
        per_repo = _per_repo(
            client_components=[proxy], server_components=[card]
        )
        links = _build_a2a_agent_cross_links(per_repo)
        assert len(links) == 1

    def test_base_host_match_between_repos(self) -> None:
        """``A2ACardResolver(base_url=host)`` matches server's ``/a2a`` card."""
        proxy = _make_proxy(
            name="weather_a2a_proxy",
            remote_url="https://weather.test",
        )
        card = _make_card("WeatherAgent", ["https://weather.test/a2a"])
        per_repo = _per_repo(
            client_components=[proxy], server_components=[card]
        )
        links = _build_a2a_agent_cross_links(per_repo)
        assert len(links) == 1

    def test_no_match_when_hosts_differ(self) -> None:
        proxy = _make_proxy(
            name="unknown_a2a_proxy",
            remote_url="https://unknown.test/a2a",
        )
        card = _make_card("WeatherAgent", ["https://weather.test/a2a"])
        per_repo = _per_repo(
            client_components=[proxy], server_components=[card]
        )
        links = _build_a2a_agent_cross_links(per_repo)
        assert links == []

    def test_non_a2a_agent_ignored(self) -> None:
        """Agents from other frameworks don't shadow A2A cross-repo matching."""
        proxy = _make_proxy(
            name="weather_a2a_proxy",
            remote_url="https://weather.test/a2a",
        )
        non_a2a_agent = AIComponent(
            name="OtherAgent",
            component_type=AIComponentType.AGENT,
            file_path="/repo-server/agent.py",
            line_number=1,
            framework="langchain",
            detection_source=DetectionSource.CODE_ANALYSIS,
            metadata={
                "agent_card": {
                    "name": "OtherAgent",
                    "endpoints": ["https://weather.test/a2a"],
                }
            },
        )
        per_repo = _per_repo(
            client_components=[proxy], server_components=[non_a2a_agent]
        )
        links = _build_a2a_agent_cross_links(per_repo)
        assert links == []

    def test_multiple_proxies_multiple_matches(self) -> None:
        proxy_a = _make_proxy(
            name="weather_a2a_proxy",
            remote_url="https://weather.test/a2a",
            line_number=10,
        )
        proxy_b = _make_proxy(
            name="finance_a2a_proxy",
            remote_url="https://finance.test/a2a",
            line_number=20,
        )
        card_a = _make_card("WeatherAgent", ["https://weather.test/a2a"])
        card_b = _make_card("FinanceAgent", ["https://finance.test/a2a"])
        per_repo = _per_repo(
            client_components=[proxy_a, proxy_b],
            server_components=[card_a, card_b],
        )
        links = _build_a2a_agent_cross_links(per_repo)
        assert len(links) == 2
        targets = {
            link.resolved_value for link in links
        }
        assert targets == {"WeatherAgent", "FinanceAgent"}

    def test_single_repo_returns_no_links_trivially(self) -> None:
        """With only one repo, cross-repo linking is vacuous."""
        proxy = _make_proxy(
            name="weather_a2a_proxy",
            remote_url="https://weather.test/a2a",
        )
        per_repo = {"/only-repo": {"components": [proxy], "relationships": []}}
        links = _build_a2a_agent_cross_links(per_repo)
        assert links == []

    def test_missing_remote_url_metadata_skipped(self) -> None:
        proxy = AIComponent(
            name="broken_a2a_proxy",
            component_type=AIComponentType.AGENT_PROXY,
            file_path="/repo-client/client.py",
            line_number=1,
            framework="a2a",
            detection_source=DetectionSource.CODE_ANALYSIS,
            metadata={
                "remote_url": "",
                "remote_verification": {
                    "status": "unresolved_url_missing",
                    "confidence": 0.3,
                    "matched_component_instance_id": "",
                    "matched_component_name": "",
                    "match_source": "",
                },
            },
        )
        card = _make_card("WeatherAgent", ["https://weather.test/a2a"])
        per_repo = _per_repo(
            client_components=[proxy], server_components=[card]
        )
        links = _build_a2a_agent_cross_links(per_repo)
        assert links == []

    def test_non_proxy_components_ignored(self) -> None:
        non_proxy = AIComponent(
            name="SomeModel",
            component_type=AIComponentType.MODEL,
            file_path="/repo-client/m.py",
            line_number=1,
            detection_source=DetectionSource.CODE_ANALYSIS,
            metadata={"remote_url": "https://weather.test/a2a"},
        )
        card = _make_card("WeatherAgent", ["https://weather.test/a2a"])
        per_repo = _per_repo(
            client_components=[non_proxy], server_components=[card]
        )
        links = _build_a2a_agent_cross_links(per_repo)
        assert links == []

    def test_card_without_endpoints_skipped(self) -> None:
        proxy = _make_proxy(
            name="weather_a2a_proxy",
            remote_url="https://weather.test/a2a",
        )
        bare_card = AIComponent(
            name="IncompleteAgent",
            component_type=AIComponentType.AGENT,
            file_path="/repo-server/.well-known/agent.json",
            line_number=0,
            framework="a2a",
            detection_source=DetectionSource.CODE_ANALYSIS,
            metadata={"agent_card": {"name": "IncompleteAgent"}},
        )
        per_repo = _per_repo(
            client_components=[proxy], server_components=[bare_card]
        )
        links = _build_a2a_agent_cross_links(per_repo)
        assert links == []

    def test_evidence_contains_endpoint_url(self) -> None:
        proxy = _make_proxy(
            name="weather_a2a_proxy",
            remote_url="https://weather.test/a2a",
        )
        card = _make_card("WeatherAgent", ["https://weather.test/a2a"])
        per_repo = _per_repo(
            client_components=[proxy], server_components=[card]
        )
        links = _build_a2a_agent_cross_links(per_repo)
        assert "https://weather.test/a2a" in links[0].evidence
        assert "WeatherAgent" in links[0].evidence
        assert "repo-client" in links[0].evidence
        assert "repo-server" in links[0].evidence


class TestIntegrationWithBuildDeterministicLinks:
    """End-to-end check that build_deterministic_cross_repo_links wires in A2A."""

    def test_a2a_link_returned_alongside_other_types(
        self, tmp_path: pytest.MonkeyPatch
    ) -> None:
        proxy = _make_proxy(
            name="weather_a2a_proxy",
            remote_url="https://weather.test/a2a",
        )
        card = _make_card("WeatherAgent", ["https://weather.test/a2a"])
        per_repo = {
            "/repo-client": {"components": [proxy], "relationships": []},
            "/repo-server": {"components": [card], "relationships": []},
        }
        links = build_deterministic_cross_repo_links(
            per_repo, ["/repo-client", "/repo-server"]
        )
        a2a_links = [
            link
            for link in links
            if link.link_type == CrossRepoLinkType.A2A_AGENT_CLIENT_SERVER
        ]
        assert len(a2a_links) == 1
        assert a2a_links[0].resolved_value == "WeatherAgent"

    def test_no_a2a_link_when_single_repo(self) -> None:
        proxy = _make_proxy(
            name="weather_a2a_proxy",
            remote_url="https://weather.test/a2a",
        )
        card = _make_card("WeatherAgent", ["https://weather.test/a2a"])
        per_repo = {
            "/only": {"components": [proxy, card], "relationships": []},
        }
        links = build_deterministic_cross_repo_links(per_repo, ["/only"])
        a2a_links = [
            link
            for link in links
            if link.link_type == CrossRepoLinkType.A2A_AGENT_CLIENT_SERVER
        ]
        assert a2a_links == []
