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

"""Tests for the offline A2A remote-agent resolver."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from aibom.cst_parser import parse_source_code
from aibom.models import AIComponent, ScanContext
from aibom.models.enums import (
    AIComponentType,
    DetectionSource,
    RelationshipType,
)
from aibom.scanners.file_cache import clear_cache
from aibom.scanners.remote_agent_resolver import (
    RemoteAgentResolver,
    _build_url_index,
    _candidates_from_calls,
    _class_defs_inheriting_a2a_client,
    _extract_url_from_call,
    _inherits_from_a2a_client,
    _match_a2a_client_call,
    _normalize_url_for_index,
    _resolve_against_index,
    _url_has_a2a_path_suffix,
    _url_is_well_known_agent_card,
)


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    clear_cache()
    yield
    clear_cache()


def _make_agent_card_component(
    name: str,
    endpoints: list[str],
    file_path: str = "/repo/.well-known/agent.json",
    line_number: int = 0,
) -> AIComponent:
    """Build a minimal AGENT component with an agent_card metadata block."""
    return AIComponent(
        name=name,
        component_type=AIComponentType.AGENT,
        file_path=file_path,
        line_number=line_number,
        framework="a2a",
        detection_source=DetectionSource.CODE_ANALYSIS,
        metadata={
            "agent_card": {
                "name": name,
                "endpoints": endpoints,
            }
        },
    )


class TestUrlNormalization:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("https://Weather.EXAMPLE.com/a2a", "https://weather.example.com/a2a"),
            ("https://x.test:443/a2a", "https://x.test/a2a"),
            ("http://x.test:80/a2a", "http://x.test/a2a"),
            ("https://x.test/a2a/", "https://x.test/a2a"),
            ("https://x.test/a2a?foo=1#bar", "https://x.test/a2a"),
            (
                "https://x.test/.well-known/agent.json",
                "https://x.test",
            ),
            (
                "https://x.test/.well-known/agent-card.json",
                "https://x.test",
            ),
            (
                "https://x.test/nested/path/.well-known/agent.json",
                "https://x.test/nested/path",
            ),
            ("https://x.test/", "https://x.test/"),
            ("https://x.test", "https://x.test"),
        ],
    )
    def test_canonical_forms(self, raw: str, expected: str) -> None:
        assert _normalize_url_for_index(raw) == expected

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "   ",
            "not a url",
            "ftp://x.test/a2a",
            "file:///etc/passwd",
            "javascript:alert(1)",
            "https://",
            "://x.test",
        ],
    )
    def test_rejects_unsupported_or_invalid(self, bad: str) -> None:
        assert _normalize_url_for_index(bad) == ""

    def test_non_string_input(self) -> None:
        assert _normalize_url_for_index(None) == ""  # type: ignore[arg-type]
        assert _normalize_url_for_index(123) == ""  # type: ignore[arg-type]

    def test_ports_other_than_default_preserved(self) -> None:
        assert (
            _normalize_url_for_index("https://x.test:8443/a2a")
            == "https://x.test:8443/a2a"
        )


class TestUrlPredicates:
    @pytest.mark.parametrize(
        "url",
        [
            "https://x.test/.well-known/agent.json",
            "HTTPS://X.test/.well-known/agent-card.json",
            "http://x.test:8080/.well-known/agent.json",
        ],
    )
    def test_well_known_match(self, url: str) -> None:
        assert _url_is_well_known_agent_card(url)

    @pytest.mark.parametrize(
        "url",
        [
            "https://x.test/a2a",
            "https://x.test/.well-known/openid-configuration",
            "https://x.test/.well-known/agent.json.bak",
            "https://x.test/agent.json",
            "",
        ],
    )
    def test_well_known_reject(self, url: str) -> None:
        assert not _url_is_well_known_agent_card(url)

    @pytest.mark.parametrize(
        "url",
        [
            "https://x.test/a2a",
            "https://x.test/a2a/",
            "https://x.test/a2a/v1",
            "https://x.test/a2a/v2/",
            "HTTP://x.test:8080/A2A",
        ],
    )
    def test_a2a_suffix_match(self, url: str) -> None:
        assert _url_has_a2a_path_suffix(url)

    @pytest.mark.parametrize(
        "url",
        [
            "https://x.test/",
            "https://x.test/a2a/forecast",
            "https://x.test/api/a2a",
            "https://x.test/a2a_v1",
            "",
        ],
    )
    def test_a2a_suffix_reject(self, url: str) -> None:
        assert not _url_has_a2a_path_suffix(url)


class TestConstructorAndInheritanceMatchers:
    @pytest.mark.parametrize(
        "qn",
        [
            "A2AClient",
            "a2a.client.A2AClient",
            "A2AAgentClient",
            "A2ACardResolver",
            "pkg.AgentClient",
            "A2ARemoteAgent",
        ],
    )
    def test_client_call_match(self, qn: str) -> None:
        assert _match_a2a_client_call(qn)

    @pytest.mark.parametrize(
        "qn",
        [
            "",
            "HTTPClient",
            "requests.Session",
            "httpx.AsyncClient",
            "a2a.server.A2AServer",
            "A2AServer",
        ],
    )
    def test_client_call_reject(self, qn: str) -> None:
        assert not _match_a2a_client_call(qn)

    def test_inheritance_match(self) -> None:
        assert _inherits_from_a2a_client(["A2AClient"])
        assert _inherits_from_a2a_client(["a2a.client.A2AClient"])
        assert _inherits_from_a2a_client(["A2ACardResolver"])
        assert _inherits_from_a2a_client(["Foo", "A2AAgentClient"])

    def test_inheritance_reject(self) -> None:
        assert not _inherits_from_a2a_client([])
        assert not _inherits_from_a2a_client([""])
        assert not _inherits_from_a2a_client(["BaseModel"])
        assert not _inherits_from_a2a_client(["A2AServer"])


class TestExtractUrlFromCall:
    def _parse_call(self, source: str):
        result = parse_source_code("x.py", source)
        calls = [c for c in result.calls if _match_a2a_client_call(c.qualified_name)]
        if not calls:
            for a in result.assignments:
                if _match_a2a_client_call(a.call.qualified_name):
                    calls.append(a.call)
        assert calls, f"no A2A client call found in:\n{source}"
        return calls[0]

    def test_kwarg_url(self) -> None:
        call = self._parse_call(
            'from a2a.client import A2AClient\n'
            'c = A2AClient(url="https://x.test/a2a")\n'
        )
        assert _extract_url_from_call(call) == "https://x.test/a2a"

    def test_kwarg_base_url(self) -> None:
        call = self._parse_call(
            'from a2a.client import A2ACardResolver\n'
            'r = A2ACardResolver(base_url="https://x.test")\n'
        )
        assert _extract_url_from_call(call) == "https://x.test"

    def test_kwarg_endpoint(self) -> None:
        call = self._parse_call(
            'from a2a.client import A2AClient\n'
            'c = A2AClient(endpoint="https://x.test/a2a/v1")\n'
        )
        assert _extract_url_from_call(call) == "https://x.test/a2a/v1"

    def test_positional_url(self) -> None:
        call = self._parse_call(
            'from a2a.client import A2AClient\n'
            'c = A2AClient("https://x.test/a2a")\n'
        )
        assert _extract_url_from_call(call) == "https://x.test/a2a"

    def test_positional_non_url_returns_none(self) -> None:
        call = self._parse_call(
            'from a2a.client import A2AClient\n'
            'c = A2AClient("some-name")\n'
        )
        assert _extract_url_from_call(call) is None

    def test_unresolved_variable_returns_none(self) -> None:
        call = self._parse_call(
            'from a2a.client import A2AClient\n'
            'u = os.getenv("AGENT_URL")\n'
            'c = A2AClient(url=u)\n'
        )
        assert _extract_url_from_call(call) is None

    def test_no_url_kwarg_returns_none(self) -> None:
        call = self._parse_call(
            'from a2a.client import A2AClient\n'
            'c = A2AClient(name="foo")\n'
        )
        assert _extract_url_from_call(call) is None

    def test_prefers_kwarg_over_positional(self) -> None:
        call = self._parse_call(
            'from a2a.client import A2AClient\n'
            'c = A2AClient("https://wrong.test/a2a", url="https://right.test/a2a")\n'
        )
        assert _extract_url_from_call(call) == "https://right.test/a2a"


class TestCandidatesFromCalls:
    def _parse(self, source: str):
        return parse_source_code("x.py", source)

    def test_direct_call_captured(self) -> None:
        r = self._parse(
            'from a2a.client import A2AClient\n'
            'A2AClient(url="https://x.test/a2a")\n'
        )
        cands = _candidates_from_calls(r)
        assert len(cands) == 1
        _call, url, ctor = cands[0]
        assert url == "https://x.test/a2a"
        assert ctor == "A2AClient"

    def test_assignment_capture(self) -> None:
        r = self._parse(
            'from a2a.client import A2AClient\n'
            'c = A2AClient(url="https://x.test/a2a")\n'
        )
        cands = _candidates_from_calls(r)
        assert len(cands) == 1

    def test_duplicate_position_not_double_counted(self) -> None:
        r = self._parse(
            'from a2a.client import A2AClient\n'
            'c = A2AClient(url="https://x.test/a2a")\n'
        )
        cands = _candidates_from_calls(r)
        assert len(cands) == 1

    def test_multiple_distinct_calls(self) -> None:
        r = self._parse(
            'from a2a.client import A2AClient\n'
            'a = A2AClient(url="https://one.test/a2a")\n'
            'b = A2AClient(url="https://two.test/a2a")\n'
        )
        cands = _candidates_from_calls(r)
        assert len(cands) == 2
        urls = {u for _, u, _ in cands}
        assert urls == {"https://one.test/a2a", "https://two.test/a2a"}

    def test_non_a2a_call_ignored(self) -> None:
        r = self._parse(
            'import requests\n'
            's = requests.Session()\n'
        )
        assert _candidates_from_calls(r) == []


class TestClassInheritance:
    def _parse(self, source: str):
        return parse_source_code("x.py", source)

    def test_class_inherits_a2a_client(self) -> None:
        r = self._parse(
            'from a2a.client import A2AClient\n'
            'class MyProxy(A2AClient):\n'
            '    pass\n'
        )
        matches = _class_defs_inheriting_a2a_client(r)
        assert len(matches) == 1
        name, line, url = matches[0]
        assert name == "MyProxy"
        assert line == 2
        assert url is None

    def test_class_inherits_with_well_known_url_in_body(self) -> None:
        r = self._parse(
            'from a2a.client import A2AClient\n'
            'class MyProxy(A2AClient):\n'
            '    CARD_URL = "https://x.test/.well-known/agent.json"\n'
        )
        matches = _class_defs_inheriting_a2a_client(r)
        assert len(matches) == 1
        _, _, url = matches[0]
        assert url == "https://x.test/.well-known/agent.json"

    def test_class_not_inheriting_ignored(self) -> None:
        r = self._parse(
            'class Foo:\n'
            '    pass\n'
        )
        assert _class_defs_inheriting_a2a_client(r) == []


class TestUrlIndex:
    def test_indexes_endpoint_urls(self) -> None:
        card = _make_agent_card_component(
            "WeatherAgent",
            endpoints=["https://weather.test/a2a", "https://weather.test/a2a/v1"],
        )
        index = _build_url_index([card])
        assert "https://weather.test/a2a" in index
        assert "https://weather.test/a2a/v1" in index

    def test_indexes_host_root_alias(self) -> None:
        card = _make_agent_card_component(
            "WeatherAgent",
            endpoints=["https://weather.test/a2a"],
        )
        index = _build_url_index([card])
        assert "https://weather.test" in index, (
            "host root should be aliased so well-known URLs match"
        )

    def test_deduplicates_same_card_under_same_key(self) -> None:
        card = _make_agent_card_component(
            "WeatherAgent",
            endpoints=["https://weather.test/a2a", "https://weather.test/a2a/"],
        )
        index = _build_url_index([card])
        assert len(index["https://weather.test/a2a"]) == 1

    def test_cards_without_endpoints_are_skipped(self) -> None:
        card = AIComponent(
            name="NoEndpointsAgent",
            component_type=AIComponentType.AGENT,
            metadata={"agent_card": {"name": "NoEndpointsAgent", "endpoints": []}},
        )
        index = _build_url_index([card])
        assert index == {}


class TestResolveAgainstIndex:
    def _index(self) -> dict:
        return _build_url_index([
            _make_agent_card_component(
                "WeatherAgent", endpoints=["https://weather.test/a2a"]
            ),
        ])

    def test_exact_match(self) -> None:
        idx = self._index()
        m = _resolve_against_index("https://weather.test/a2a", idx)
        assert m is not None and m.name == "WeatherAgent"

    def test_trailing_slash_tolerant(self) -> None:
        idx = self._index()
        m = _resolve_against_index("https://weather.test/a2a/", idx)
        assert m is not None and m.name == "WeatherAgent"

    def test_host_root_fallback(self) -> None:
        idx = self._index()
        m = _resolve_against_index(
            "https://weather.test/.well-known/agent.json", idx
        )
        assert m is not None and m.name == "WeatherAgent"

    def test_base_url_fallback_to_host(self) -> None:
        idx = self._index()
        m = _resolve_against_index("https://weather.test", idx)
        assert m is not None and m.name == "WeatherAgent"

    def test_no_match_returns_none(self) -> None:
        idx = self._index()
        assert _resolve_against_index("https://other.test/a2a", idx) is None

    def test_invalid_url_returns_none(self) -> None:
        idx = self._index()
        assert _resolve_against_index("not a url", idx) is None


class TestEndToEndScan:
    def _setup_repo(self, tmp_path: Path) -> Path:
        (tmp_path / ".well-known").mkdir()
        (tmp_path / ".well-known" / "agent.json").write_text(
            json.dumps({
                "name": "WeatherAgent",
                "description": "Weather forecasts",
                "version": "1.0.0",
                "supportedInterfaces": [
                    {"url": "https://weather.test/a2a", "type": "jsonrpc"}
                ],
            })
        )
        return tmp_path

    def test_verified_client_constructor(self, tmp_path: Path) -> None:
        self._setup_repo(tmp_path)
        (tmp_path / "client.py").write_text(
            'from a2a.client import A2AClient\n'
            'c = A2AClient(url="https://weather.test/a2a")\n'
        )
        proxies, rels = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert len(proxies) == 1
        p = proxies[0]
        assert p.component_type == AIComponentType.AGENT_PROXY
        assert p.framework == "a2a"
        verification = p.metadata["remote_verification"]
        assert verification["status"] == "verified_local_card"
        assert verification["confidence"] == 1.0
        assert verification["matched_component_name"] == "WeatherAgent"
        assert p.metadata["detection_reason"] == "client_constructor_call"
        assert len(rels) == 1
        assert rels[0].relationship_type == RelationshipType.INVOKES_A2A_AGENT
        assert rels[0].target_name == "WeatherAgent"
        assert rels[0].source_type == AIComponentType.AGENT_PROXY
        assert rels[0].target_type == AIComponentType.AGENT

    def test_unverified_when_host_unknown(self, tmp_path: Path) -> None:
        self._setup_repo(tmp_path)
        (tmp_path / "client.py").write_text(
            'from a2a.client import A2AClient\n'
            'c = A2AClient(url="https://unknown.test/a2a")\n'
        )
        proxies, rels = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert len(proxies) == 1
        verification = proxies[0].metadata["remote_verification"]
        assert verification["status"] == "unverified_url_pattern"
        assert verification["confidence"] == 0.5
        assert verification["matched_component_instance_id"] == ""
        assert rels == []

    def test_unresolved_url_missing_for_class_inheritance_alone(
        self, tmp_path: Path
    ) -> None:
        self._setup_repo(tmp_path)
        (tmp_path / "client.py").write_text(textwrap.dedent('''
            from a2a.client import A2AClient

            class MyProxy(A2AClient):
                pass
        ''').lstrip())
        proxies, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert len(proxies) == 1
        p = proxies[0]
        assert p.metadata["remote_verification"]["status"] == "unresolved_url_missing"
        assert p.metadata["remote_verification"]["confidence"] == 0.3
        assert p.metadata["class_name"] == "MyProxy"
        assert p.metadata["remote_url"] == ""

    def test_card_resolver_base_url_matches_card(self, tmp_path: Path) -> None:
        self._setup_repo(tmp_path)
        (tmp_path / "resolve.py").write_text(
            'from a2a.client import A2ACardResolver\n'
            'r = A2ACardResolver(base_url="https://weather.test")\n'
        )
        proxies, rels = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert len(proxies) == 1
        assert (
            proxies[0].metadata["remote_verification"]["status"]
            == "verified_local_card"
        )
        assert len(rels) == 1

    def test_well_known_string_literal_inside_class(self, tmp_path: Path) -> None:
        """Well-known URL in a class body emits a proxy even without inheritance.

        Module-level string literals are deliberately ignored (the CST
        parser restricts ``protocol_strings`` to class scope to avoid
        noise). Real A2A client code wraps the URL inside a class
        anyway, so the ``well_known_url_literal`` signal is scoped to
        class bodies.
        """
        self._setup_repo(tmp_path)
        (tmp_path / "config.py").write_text(textwrap.dedent('''
            class WeatherConfig:
                CARD_URL = "https://weather.test/.well-known/agent.json"
        ''').lstrip())
        proxies, rels = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        well_known_proxies = [
            p
            for p in proxies
            if p.metadata.get("detection_reason") == "well_known_url_literal"
        ]
        assert len(well_known_proxies) == 1
        p = well_known_proxies[0]
        assert (
            p.metadata["remote_verification"]["status"] == "verified_local_card"
        )
        assert len(rels) == 1

    def test_module_level_well_known_literal_ignored(self, tmp_path: Path) -> None:
        """Module-level config strings should NOT produce proxies.

        The CST parser scopes ``protocol_strings`` to class bodies to
        keep arbitrary module-level constants (often just config keys
        or docstrings) out of the signal. This asserts that contract.
        """
        self._setup_repo(tmp_path)
        (tmp_path / "config.py").write_text(
            'CARD_URL = "https://weather.test/.well-known/agent.json"\n'
        )
        proxies, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert proxies == []

    def test_plain_a2a_suffix_literal_does_not_emit_standalone(
        self, tmp_path: Path
    ) -> None:
        """``/a2a`` URL literal without constructor/base-class is ignored.

        Rationale: ``/a2a`` alone is ambiguous (could be any path); only
        ``.well-known/agent(-card).json`` is specific enough to emit on
        its own. Constructor/base-class signals handle the rest.
        """
        self._setup_repo(tmp_path)
        (tmp_path / "config.py").write_text(
            'SOME_URL = "https://not-even-related.test/a2a"\n'
        )
        proxies, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert proxies == []

    def test_multiple_proxies_same_file(self, tmp_path: Path) -> None:
        self._setup_repo(tmp_path)
        (tmp_path / "client.py").write_text(textwrap.dedent('''
            from a2a.client import A2AClient

            known = A2AClient(url="https://weather.test/a2a")
            unknown = A2AClient(url="https://unknown.test/a2a")
        ''').lstrip())
        proxies, rels = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert len(proxies) == 2
        statuses = {
            p.metadata["remote_verification"]["status"] for p in proxies
        }
        assert statuses == {"verified_local_card", "unverified_url_pattern"}
        assert len(rels) == 1

    def test_deduplicates_by_file_line_and_url(self, tmp_path: Path) -> None:
        self._setup_repo(tmp_path)
        (tmp_path / "client.py").write_text(
            'from a2a.client import A2AClient\n'
            'c = A2AClient(url="https://weather.test/a2a")\n'
        )
        proxies1, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        proxies2, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert len(proxies1) == 1
        assert len(proxies2) == 1

    def test_emits_no_components_in_empty_tree(self, tmp_path: Path) -> None:
        (tmp_path / "misc.py").write_text("x = 1\n")
        proxies, rels = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert proxies == []
        assert rels == []

    def test_needs_agentic_false(self, tmp_path: Path) -> None:
        """AGENT_PROXY components don't trigger the LLM agentic pipeline.

        They are definitionally non-agents — the proxy pattern is an
        offline structural fact, not a classification the LLM needs to
        confirm.
        """
        self._setup_repo(tmp_path)
        (tmp_path / "client.py").write_text(
            'from a2a.client import A2AClient\n'
            'c = A2AClient(url="https://weather.test/a2a")\n'
        )
        proxies, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert proxies[0].needs_agentic is False

    def test_detection_source_is_code_analysis(self, tmp_path: Path) -> None:
        self._setup_repo(tmp_path)
        (tmp_path / "client.py").write_text(
            'from a2a.client import A2AClient\n'
            'c = A2AClient(url="https://weather.test/a2a")\n'
        )
        proxies, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert proxies[0].detection_source == DetectionSource.CODE_ANALYSIS


class TestSupportsAndName:
    def test_scanner_name(self) -> None:
        assert RemoteAgentResolver.name == "remote_agent_resolver"

    def test_supports_always_true(self) -> None:
        ctx = ScanContext(paths=["/nonexistent"])
        assert RemoteAgentResolver().supports(ctx) is True


class TestRobustness:
    def test_invalid_python_file_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "bad.py").write_text("def broken(:::\n")
        (tmp_path / "good.py").write_text(
            'from a2a.client import A2AClient\n'
            'c = A2AClient(url="https://x.test/a2a")\n'
        )
        proxies, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert len(proxies) == 1
        assert proxies[0].metadata["remote_url"] == "https://x.test/a2a"

    def test_oversized_python_file_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Files exceeding the resolver's size cap are skipped.

        We monkeypatch the cap to a tiny value rather than writing a
        real multi-MB file — that way we exercise the size guard
        without forcing upstream scanners (A2ADetector) to CST-parse a
        huge file, which would hang the test. The cap itself exists
        so that one enormous generated file cannot DoS the scanner.
        """
        from aibom.scanners import remote_agent_resolver as r

        monkeypatch.setattr(r, "_MAX_PY_FILE_SIZE_BYTES", 32)
        (tmp_path / "big.py").write_text(
            'from a2a.client import A2AClient\n'
            'c = A2AClient(url="https://x.test/a2a")\n'
        )
        proxies, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert proxies == []

    def test_non_python_files_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text(
            'A2AClient(url="https://x.test/a2a")'
        )
        proxies, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert proxies == []

    def test_preselect_skips_files_without_hints(self, tmp_path: Path) -> None:
        """Files with no A2A-related substring never hit the CST parser."""
        (tmp_path / "boring.py").write_text(
            "class Foo:\n    def bar(self): return 1\n"
        )
        proxies, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert proxies == []


# ---------------------------------------------------------------------------
# HTTP/SSE fallback: code that delegates a request loop to a remote
# LLM/agent service over HTTP/SSE, when no A2A-specific marker is present.
# All fixtures use synthetic class/module names — none map to any real
# production class.
# ---------------------------------------------------------------------------


class TestSdkFallback:
    """Tests for the remote agent-runtime SDK ``AGENT_PROXY`` fallback path.

    The fallback promotes a class to ``AGENT_PROXY`` only when it calls
    into an SDK where the *server* runs the agentic loop (OpenAI
    Assistants, LangGraph Cloud, AWS Bedrock Agents, Vertex Reasoning
    Engines, or user-extended signatures from ``.aibom.yaml``). Plain
    chat-completion calls are deliberately NOT promoted — the local
    caller runs the loop in that case.
    """

    def _write_module(self, tmp_path: Path, name: str, body: str) -> Path:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
        return p

    def test_openai_assistants_run_create_is_promoted(
        self, tmp_path: Path
    ) -> None:
        """A class that calls ``client.beta.threads.runs.create`` is a proxy.

        The OpenAI Assistants API runs the agentic loop on OpenAI's side
        (tool routing, iteration, state); the local class is a thin
        client. It must surface as ``AGENT_PROXY`` with ``sdk_id``
        metadata so the Phase-5 prompt can confirm and Phase-6 can
        re-verify the citation.
        """
        self._write_module(
            tmp_path,
            "assistants_wrapper.py",
            '''
            from openai import OpenAI


            class AssistantsWrapper:
                """Delegates an agent run to the OpenAI Assistants API."""

                def __init__(self):
                    self.client = OpenAI()

                def run(self, thread_id: str, assistant_id: str):
                    return self.client.beta.threads.runs.create(
                        thread_id=thread_id,
                        assistant_id=assistant_id,
                    )
            ''',
        )
        proxies, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert len(proxies) == 1
        p = proxies[0]
        assert p.component_type == AIComponentType.AGENT_PROXY
        assert p.metadata["detection_reason"] == "remote_agent_sdk_call"
        assert p.metadata["sdk_id"] == "openai.assistants"
        assert p.metadata["class_name"] == "AssistantsWrapper"
        assert p.metadata["call_qualified_name"].endswith(
            "client.beta.threads.runs.create"
        )
        assert p.metadata["remote_url"] == ""
        assert p.framework == "openai.assistants"
        verification = p.metadata["remote_verification"]
        assert verification["status"] == "unresolved_url_missing"
        assert verification["confidence"] == 0.7

    def test_langgraph_remote_graph_is_promoted(
        self, tmp_path: Path
    ) -> None:
        """LangGraph ``RemoteGraph`` is a first-class remote-agent SDK.

        We accept both the ``RemoteGraph(url=…).invoke`` instance-method
        form and the qualified class constructor import. The signature's
        ``import_substrings`` gate (``langgraph``) ensures the bare
        ``RemoteGraph`` name doesn't match user-defined classes with
        the same name in unrelated packages.
        """
        self._write_module(
            tmp_path,
            "router.py",
            '''
            from langgraph.pregel.remote import RemoteGraph


            class RouterProxy:
                def __init__(self, url: str):
                    self.graph = RemoteGraph("agent", url=url)

                def run(self, inputs):
                    return self.graph.invoke(inputs)
            ''',
        )
        proxies, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert len(proxies) == 1
        p = proxies[0]
        assert p.metadata["sdk_id"] == "langgraph.remote_graph"
        assert p.metadata["class_name"] == "RouterProxy"

    def test_bedrock_classic_invoke_agent_is_promoted(
        self, tmp_path: Path
    ) -> None:
        """Classic AWS Bedrock Agents Runtime (``bedrock-agent-runtime``).

        The ``bedrock-agent-runtime`` boto3 service hosts AWS-managed
        agents with action groups and knowledge bases; the loop runs
        on AWS, so the local class is a proxy.
        """
        self._write_module(
            tmp_path,
            "bedrock_classic.py",
            '''
            import boto3


            class BedrockClassicProxy:
                def __init__(self):
                    self.client = boto3.client("bedrock-agent-runtime")

                def run(self, agent_id, alias, session_id, prompt):
                    return self.client.invoke_agent(
                        agentId=agent_id,
                        agentAliasId=alias,
                        sessionId=session_id,
                        inputText=prompt,
                    )
            ''',
        )
        proxies, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert len(proxies) == 1
        p = proxies[0]
        assert p.metadata["sdk_id"] == "aws.bedrock_agents"
        assert p.metadata["class_name"] == "BedrockClassicProxy"
        assert p.metadata["call_qualified_name"].endswith(
            ".invoke_agent"
        )

    def test_bedrock_agentcore_runtime_is_promoted(
        self, tmp_path: Path
    ) -> None:
        """AWS Bedrock AgentCore Runtime (``bedrock-agentcore``).

        The ``bedrock-agentcore`` boto3 service hosts customer-built
        agents (LangGraph, CrewAI, Strands, custom Python) inside
        AWS-managed sandboxes; the loop still runs server-side, so a
        local ``client.invoke_agent_runtime(agentRuntimeArn=…)`` caller
        is a proxy. This must produce a *separate* SDK id from the
        classic Bedrock Agents service.
        """
        self._write_module(
            tmp_path,
            "agentcore_wrapper.py",
            '''
            import boto3
            import json


            class AgentCoreRuntimeProxy:
                def __init__(self, runtime_arn: str):
                    self.client = boto3.client("bedrock-agentcore")
                    self.runtime_arn = runtime_arn

                def run(self, session_id: str, prompt: str):
                    payload = json.dumps({"prompt": prompt}).encode()
                    return self.client.invoke_agent_runtime(
                        agentRuntimeArn=self.runtime_arn,
                        runtimeSessionId=session_id,
                        payload=payload,
                    )
            ''',
        )
        proxies, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert len(proxies) == 1
        p = proxies[0]
        assert p.metadata["sdk_id"] == "aws.bedrock_agentcore_runtime"
        assert p.metadata["class_name"] == "AgentCoreRuntimeProxy"
        assert p.metadata["call_qualified_name"].endswith(
            ".invoke_agent_runtime"
        )
        assert p.framework == "aws.bedrock_agentcore_runtime"

    def test_plain_chat_completions_is_not_promoted(
        self, tmp_path: Path
    ) -> None:
        """``chat.completions.create`` is NOT a remote agent SDK call.

        This is the most important false-positive guard: a plain OpenAI
        chat-completion client runs the agent loop locally (if at all).
        It must not be promoted just because it imports ``openai`` and
        streams tokens. Only *Assistants*-style endpoints where the
        server owns the loop count.
        """
        self._write_module(
            tmp_path,
            "plain_llm.py",
            '''
            from openai import OpenAI


            class PlainCompletions:
                def __init__(self):
                    self.client = OpenAI()

                def ask(self, prompt: str):
                    return self.client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": prompt}],
                        stream=True,
                    )
            ''',
        )
        proxies, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert proxies == []

    def test_plain_rest_client_is_not_promoted(self, tmp_path: Path) -> None:
        """An unrelated HTTP REST client never matches any SDK signature."""
        self._write_module(
            tmp_path,
            "user_client.py",
            '''
            import httpx


            class UserProfileClient:
                URL = "https://api.example.test/api/users"

                def fetch(self, uid):
                    with httpx.Client() as c:
                        return c.get(f"{self.URL}/{uid}").json()
            ''',
        )
        proxies, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert proxies == []

    def test_streaming_log_tailer_is_not_promoted(
        self, tmp_path: Path
    ) -> None:
        """Streaming arbitrary data over HTTP does not match any SDK.

        Under the previous path-heuristic implementation this class could
        have been flagged because of ``iter_lines`` + HTTP. The new
        SDK-based matcher requires an explicit, documented SDK call, so
        a log-tailer never qualifies.
        """
        self._write_module(
            tmp_path,
            "log_tail.py",
            '''
            import httpx


            class StreamingLogTailer:
                URL = "https://telemetry.example.test/v1/logs/stream"

                def tail(self):
                    with httpx.Client() as c:
                        with c.stream("GET", self.URL) as r:
                            for line in r.iter_lines():
                                yield line
            ''',
        )
        proxies, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert proxies == []

    def test_module_scope_sdk_call_is_ignored(self, tmp_path: Path) -> None:
        """Standalone scripts that call an SDK once are not proxy classes.

        A module-level ``client.beta.threads.runs.create(...)`` call has
        no enclosing class, so there is nothing to tag as a proxy.
        """
        self._write_module(
            tmp_path,
            "script.py",
            '''
            from openai import OpenAI


            client = OpenAI()
            run = client.beta.threads.runs.create(
                thread_id="t1",
                assistant_id="a1",
            )
            ''',
        )
        proxies, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert proxies == []

    def test_import_substring_gate_rejects_wrong_package(
        self, tmp_path: Path
    ) -> None:
        """The ``import_substrings`` gate prevents short-name collisions.

        An unrelated class also named ``RemoteGraph`` but imported from
        ``my.util`` (no ``langgraph`` import anywhere) must NOT match the
        LangGraph SDK signature even though the bare class name is
        identical.
        """
        self._write_module(
            tmp_path,
            "lookalike.py",
            '''
            from my.util import RemoteGraph


            class LookAlike:
                def __init__(self):
                    self.graph = RemoteGraph()

                def run(self, x):
                    return self.graph.invoke(x)
            ''',
        )
        proxies, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert proxies == []

    def test_a2a_match_takes_precedence_over_sdk(
        self, tmp_path: Path
    ) -> None:
        """A class matching both A2A and an SDK is only emitted via A2A.

        The A2A path attaches the proxy to the client-constructor line,
        not the class definition, and the SDK pass must skip any class
        whose body contains any A2A-tagged line.
        """
        self._write_module(
            tmp_path,
            "dual.py",
            '''
            from a2a.client import A2AClient
            from openai import OpenAI


            class DualSignalProxy:
                def __init__(self):
                    self.a2a = A2AClient(url="https://dual.test/a2a")
                    self.openai = OpenAI()

                def run(self, thread_id, assistant_id):
                    return self.openai.beta.threads.runs.create(
                        thread_id=thread_id,
                        assistant_id=assistant_id,
                    )
            ''',
        )
        proxies, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert len(proxies) == 1
        p = proxies[0]
        assert p.metadata["detection_reason"] == "client_constructor_call"
        assert p.metadata.get("sdk_id", "") == ""
        assert p.framework != "openai.assistants"

    def test_temporal_workflow_anti_pattern_suppresses_proxy(
        self, tmp_path: Path
    ) -> None:
        """A Temporal workflow that would otherwise match SDK is skipped."""
        self._write_module(
            tmp_path,
            "workflows.py",
            '''
            from openai import OpenAI
            from temporalio import workflow


            @workflow.defn
            class StreamingWorkflow:
                def __init__(self):
                    self.client = OpenAI()

                @workflow.run
                async def run(self, thread_id, assistant_id):
                    return self.client.beta.threads.runs.create(
                        thread_id=thread_id,
                        assistant_id=assistant_id,
                    )
            ''',
        )
        proxies, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert proxies == []

    def test_pydantic_basemodel_anti_pattern_suppresses_proxy(
        self, tmp_path: Path
    ) -> None:
        """A pydantic BaseModel subclass is never promoted."""
        self._write_module(
            tmp_path,
            "schemas.py",
            '''
            from openai import OpenAI
            from pydantic import BaseModel


            class RequestEnvelope(BaseModel):
                thread_id: str

                def run(self, client: OpenAI, assistant_id: str):
                    return client.beta.threads.runs.create(
                        thread_id=self.thread_id,
                        assistant_id=assistant_id,
                    )
            ''',
        )
        proxies, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert proxies == []

    def test_framework_agent_base_class_is_left_to_other_scanners(
        self, tmp_path: Path
    ) -> None:
        """Framework-agent subclasses are the KB/config scanners' job."""
        self._write_module(
            tmp_path,
            "framework.py",
            '''
            from openai import OpenAI
            from langchain.agents import BaseSingleActionAgent


            class FrameworkAgent(BaseSingleActionAgent):
                def __init__(self):
                    self.client = OpenAI()

                def run(self, thread_id, assistant_id):
                    return self.client.beta.threads.runs.create(
                        thread_id=thread_id,
                        assistant_id=assistant_id,
                    )
            ''',
        )
        proxies, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert proxies == []

    def test_class_line_number_attachment(self, tmp_path: Path) -> None:
        """The emitted proxy is attached to the class definition line."""
        mod = self._write_module(
            tmp_path,
            "attach.py",
            '''
            from openai import OpenAI


            class AttachmentPoint:
                def __init__(self):
                    self.client = OpenAI()

                def run(self, thread_id, assistant_id):
                    return self.client.beta.threads.runs.create(
                        thread_id=thread_id,
                        assistant_id=assistant_id,
                    )
            ''',
        )
        proxies, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert len(proxies) == 1
        p = proxies[0]
        source_lines = mod.read_text(encoding="utf-8").splitlines()
        class_line_idx = next(
            i for i, ln in enumerate(source_lines, start=1)
            if ln.startswith("class AttachmentPoint")
        )
        assert p.line_number == class_line_idx

    def test_one_proxy_per_class_sdk_pair(self, tmp_path: Path) -> None:
        """Multiple SDK calls in one class still produce a single proxy."""
        self._write_module(
            tmp_path,
            "multi.py",
            '''
            from openai import OpenAI


            class MultiCall:
                def __init__(self):
                    self.client = OpenAI()

                def a(self):
                    return self.client.beta.threads.runs.create(
                        thread_id="t", assistant_id="a"
                    )

                def b(self):
                    return self.client.beta.threads.runs.create(
                        thread_id="u", assistant_id="b"
                    )
            ''',
        )
        proxies, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert len(proxies) == 1

    def test_oversized_file_skipped_for_sdk_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The 2 MiB size cap applies to the SDK pass too."""
        from aibom.scanners import remote_agent_resolver as r

        monkeypatch.setattr(r, "_MAX_PY_FILE_SIZE_BYTES", 64)
        self._write_module(
            tmp_path,
            "big.py",
            '''
            from openai import OpenAI


            class BigSdkProxy:
                def __init__(self):
                    self.client = OpenAI()

                def run(self):
                    return self.client.beta.threads.runs.create(
                        thread_id="t", assistant_id="a"
                    )
            ''',
        )
        proxies, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert proxies == []

    def test_user_defined_sdk_via_aibom_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Custom SDK signatures from ``.aibom.yaml`` are respected.

        This exercises the catalog extensibility contract: a signature
        added under ``agent_signatures.remote_agent_sdks`` with no
        ``import_substrings`` gates purely on the call suffix, which is
        useful for internal, privately distributed agent-runtime SDKs.
        """
        from aibom import agent_signatures as sigs_mod
        from aibom.agent_signatures import (
            AgentSignatureCatalog,
            RemoteAgentSdkSignature,
            default_catalog,
        )

        base = default_catalog()
        custom = AgentSignatureCatalog(
            frameworks=list(base.frameworks),
            protocols=list(base.protocols),
            anti_patterns=list(base.anti_patterns),
            remote_agent_sdks=list(base.remote_agent_sdks)
            + [
                RemoteAgentSdkSignature(
                    id="custom.internal_runtime",
                    sdk_name="Custom Internal Agent Runtime",
                    import_substrings=(),
                    call_qualified_suffixes=("run_remote_agent",),
                    description="Project-internal remote agent runtime.",
                )
            ],
            verification_policy=base.verification_policy,
        )

        monkeypatch.setattr(
            sigs_mod, "resolve_catalog", lambda overrides=None: custom
        )
        from aibom.scanners import remote_agent_resolver as resolver_mod

        monkeypatch.setattr(
            resolver_mod, "resolve_catalog", lambda: custom
        )

        self._write_module(
            tmp_path,
            "internal.py",
            '''
            class InternalRuntimeProxy:
                def __init__(self, runtime):
                    self.runtime = runtime

                def run(self, inputs):
                    return self.runtime.run_remote_agent(inputs)
            ''',
        )
        proxies, _ = RemoteAgentResolver().scan(
            ScanContext(paths=[str(tmp_path)])
        )
        assert len(proxies) == 1
        p = proxies[0]
        assert p.metadata["sdk_id"] == "custom.internal_runtime"
        assert p.metadata["class_name"] == "InternalRuntimeProxy"
        assert p.framework == "custom.internal_runtime"
