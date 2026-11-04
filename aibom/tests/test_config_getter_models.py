# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Model / endpoint extraction from config-getter defaults and conditionals."""

from __future__ import annotations

import libcst as cst

from aibom.cst_parser import (
    _extract_argument_value,
    literal_branches,
    parse_source_code,
)
from aibom.models import AIComponentType
from aibom.scanners.kb_enrichment_scanner import (
    _detect_config_getter_defaults,
    _detect_model_kwargs,
)
_PROVIDER_CONFIG = '''
class ProviderConfig(ServiceConfig):
    def _configure(self) -> None:
        shared_api_base = self.module_config.api_base
        self.api_base = self._get_str(
            "api_base",
            shared_api_base if shared_api_base is not None else "https://llm.example",
        )
        shared_model = self.module_config.model
        self.model = self._get_str(
            "model",
            shared_model if shared_model is not None else "gpt-4o",
        )
        self.fallback = self._get_str("model_name", "claude-3-5-sonnet-20241022")
        self.embedding_model = self._get_str(
            "embedding_model",
            (shared_model if shared_model is not None else "text-embedding-3-small"),
        )
        self.style = self._get_enum("response_format", ["json", "text"], "text")
        self.api_key = self._get_str("api_key", "")
        self.tenant = self._get_str("model", None if self.app_login else "common")
'''


def _getter_method(args: str) -> str:
    return (
        "class C:\n"
        "    def f(self):\n"
        f"        self.value = self._get_str({args})\n"
    )


def _getter_components(source: str) -> dict[AIComponentType, set[str]]:
    result = parse_source_code("provider.py", source)
    found: dict[AIComponentType, set[str]] = {}
    for comp in _detect_config_getter_defaults(result):
        found.setdefault(comp.component_type, set()).add(comp.name)
    return found


class TestIfExpExtraction:
    def test_both_branches_are_kept(self) -> None:
        node = cst.parse_expression('shared if shared is not None else "gpt-4o"')
        value = _extract_argument_value(node)
        assert value == {"_ifexp": ["VARIABLE:shared", "gpt-4o"]}
        assert literal_branches(value) == ["gpt-4o"]

    def test_nested_conditionals_flatten(self) -> None:
        node = cst.parse_expression('"a-1" if x else ("b-2" if y else z)')
        assert literal_branches(_extract_argument_value(node)) == ["a-1", "b-2"]

    def test_non_literals_yield_nothing(self) -> None:
        assert literal_branches("VARIABLE:model") == []
        assert literal_branches("ATTRIBUTE:self.model") == []
        assert literal_branches({"_call": "os.getenv", "_args": ["MODEL"]}) == []
        assert literal_branches("gpt-4o") == ["gpt-4o"]


class TestConfigGetterDefaults:
    def test_model_endpoint_and_embedding_defaults(self) -> None:
        found = _getter_components(_PROVIDER_CONFIG)
        assert found[AIComponentType.MODEL] == {
            "gpt-4o",
            "claude-3-5-sonnet-20241022",
        }
        assert found[AIComponentType.LLM_ENDPOINT] == {"https://llm.example"}
        assert found[AIComponentType.EMBEDDING] == {"text-embedding-3-small"}

    def test_unrelated_keys_and_non_model_literals_are_ignored(self) -> None:
        found = _getter_components(_PROVIDER_CONFIG)
        names = set().union(*found.values())
        assert "text" not in names
        assert "common" not in names
        assert "" not in names

    def test_unhinted_provider_name_is_not_a_model(self) -> None:
        source = _getter_method('"model", "local"')
        assert _getter_components(source) == {}

    def test_non_getter_calls_are_ignored(self) -> None:
        source = 'value = settings.lookup("model", "gpt-4o")\n'
        assert _getter_components(source) == {}

    def test_model_detected_without_registry_hit(self) -> None:
        source = _getter_method('"model", "gemini-pro"')
        found = _getter_components(source)
        assert found[AIComponentType.MODEL] == {"gemini-pro"}


class TestModelKwargConditional:
    def test_both_registry_known_branches_are_emitted(self) -> None:
        source = (
            "from langchain_openai import ChatOpenAI\n"
            'llm = ChatOpenAI(model="gpt-4o" if fast else "gpt-4-turbo")\n'
        )
        result = parse_source_code("app.py", source)
        names = {
            c.name
            for c in _detect_model_kwargs(result)
            if c.component_type == AIComponentType.MODEL
        }
        assert names == {"gpt-4o", "gpt-4-turbo"}
