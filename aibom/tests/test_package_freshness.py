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

import json
from datetime import datetime, timezone

import duckdb
import httpx
import pytest
import respx
from typer.main import get_command

from aibom.cli import app
from aibom.models import AIComponent
from aibom.models.enums import AIComponentType
from aibom.package_freshness import (
    enrich_analysis_outputs,
    enrich_components,
    resolve_freshness_url,
)
from aibom.package_registry import (
    package_coordinate,
    read_package_snapshots,
    resolve_package_catalog_path,
)

FRESHNESS_URL = "https://freshness.example.test/packages"


def _catalog(tmp_path, *, snapshot_at: str = "2026-07-01T00:00:00Z"):
    path = tmp_path / "catalog.duckdb"
    connection = duckdb.connect(str(path))
    try:
        connection.execute(
            """
            CREATE TABLE package_catalog (
                package_name VARCHAR NOT NULL,
                ecosystem VARCHAR NOT NULL,
                liveness_status VARCHAR,
                liveness_snapshot_at TIMESTAMP,
                liveness_certification JSON
            )
            """
        )
        connection.execute(
            """
            INSERT INTO package_catalog
            VALUES (?, ?, ?, CAST(? AS TIMESTAMP), CAST(? AS JSON))
            """,
            [
                "langchain",
                "pypi",
                "maintained",
                snapshot_at,
                '{"source_count": 2}',
            ],
        )
    finally:
        connection.close()
    return path


def _dependency(name: str = "langchain") -> AIComponent:
    return AIComponent(
        name=name,
        component_type=AIComponentType.DEPENDENCY,
        metadata={"ecosystem": "pypi"},
    )


def test_registry_reads_only_requested_package_snapshots(tmp_path):
    path = _catalog(tmp_path)
    requested = package_coordinate("pypi", "langchain")
    missing = package_coordinate("pypi", "unrelated-package")

    snapshots = read_package_snapshots(path, [requested, missing])

    assert set(snapshots) == {requested}
    assert snapshots[requested].liveness_status == "maintained"
    assert snapshots[requested].liveness_snapshot_at == "2026-07-01T00:00:00Z"
    assert snapshots[requested].certification == {"source_count": 2}


def test_snapshot_only_enrichment_never_calls_network(tmp_path):
    path = _catalog(tmp_path)
    component = _dependency()

    with respx.mock:
        enrich_components(
            [component],
            db_path=path,
            freshness_url=FRESHNESS_URL,
            no_network=True,
            now=datetime(2026, 7, 24, tzinfo=timezone.utc),
        )

    assert component.metadata["liveness_status"] == "maintained"
    assert component.metadata["liveness_snapshot_at"] == (
        "2026-07-01T00:00:00Z"
    )
    assert component.metadata["liveness_certification"] == {"source_count": 2}
    assert "as_of" not in component.metadata


def test_liveness_only_snapshot_never_calls_network(tmp_path):
    path = _catalog(tmp_path)
    component = _dependency()

    with respx.mock:
        enrich_components(
            [component],
            db_path=path,
            freshness_url=FRESHNESS_URL,
            liveness_only_snapshot=True,
            now=datetime(2026, 7, 24, tzinfo=timezone.utc),
        )

    assert component.metadata["liveness_status"] == "maintained"
    assert "as_of" not in component.metadata


@respx.mock
def test_stale_snapshot_uses_anonymous_live_delta_for_bom_packages(tmp_path):
    path = _catalog(tmp_path)
    component = _dependency()
    unrelated = _dependency("unrelated-package")
    route = respx.post(FRESHNESS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "snapshot_version": "build-2",
                "as_of": "2026-07-24T00:00:00Z",
                "signals": [
                    {
                        "snapshot_version": "build-2",
                        "ecosystem": "pypi",
                        "package_name": "langchain",
                        "liveness_status": "stale",
                        "observed_at": "2026-07-23T12:00:00Z",
                        "source": "registry",
                        "certification": {"source_count": 3},
                    }
                ],
            },
        )
    )

    enrich_components(
        [component, unrelated],
        db_path=path,
        freshness_url=FRESHNESS_URL,
        now=datetime(2026, 7, 24, tzinfo=timezone.utc),
    )

    assert route.called
    request = route.calls.last.request
    assert json.loads(request.content) == {
        "snapshot_at": "2026-07-01T00:00:00Z",
        "packages": [{"ecosystem": "pypi", "name": "langchain"}],
    }
    assert "authorization" not in request.headers
    assert component.metadata["liveness_status"] == "stale"
    assert component.metadata["as_of"] == "2026-07-24T00:00:00Z"
    assert component.metadata["liveness_observed_at"] == (
        "2026-07-23T12:00:00Z"
    )
    assert "liveness_status" not in unrelated.metadata


def test_fresh_snapshot_does_not_call_network(tmp_path):
    path = _catalog(tmp_path, snapshot_at="2026-07-23T00:00:00Z")
    component = _dependency()

    with respx.mock:
        enrich_components(
            [component],
            db_path=path,
            freshness_url=FRESHNESS_URL,
            now=datetime(2026, 7, 24, tzinfo=timezone.utc),
        )

    assert component.metadata["liveness_status"] == "maintained"
    assert "as_of" not in component.metadata


@respx.mock
def test_rate_limit_falls_back_to_snapshot(tmp_path, caplog):
    path = _catalog(tmp_path)
    component = _dependency()
    respx.post(FRESHNESS_URL).mock(
        return_value=httpx.Response(429, headers={"Retry-After": "60"})
    )

    enrich_components(
        [component],
        db_path=path,
        freshness_url=FRESHNESS_URL,
        now=datetime(2026, 7, 24, tzinfo=timezone.utc),
    )

    assert component.metadata["liveness_status"] == "maintained"
    assert "as_of" not in component.metadata
    assert "rate limited" in caplog.text


@respx.mock
def test_timeout_falls_back_to_snapshot(tmp_path, caplog):
    path = _catalog(tmp_path)
    component = _dependency()
    respx.post(FRESHNESS_URL).mock(
        side_effect=httpx.ReadTimeout("synthetic timeout")
    )

    enrich_components(
        [component],
        db_path=path,
        freshness_url=FRESHNESS_URL,
        now=datetime(2026, 7, 24, tzinfo=timezone.utc),
    )

    assert component.metadata["liveness_status"] == "maintained"
    assert "as_of" not in component.metadata
    assert "using snapshot data" in caplog.text


def test_legacy_catalog_without_package_table_is_a_no_op(tmp_path):
    path = tmp_path / "legacy.duckdb"
    connection = duckdb.connect(str(path))
    connection.execute("CREATE TABLE component_catalog (id VARCHAR)")
    connection.close()
    component = _dependency()

    with respx.mock:
        enrich_components(
            [component],
            db_path=path,
            freshness_url=FRESHNESS_URL,
            now=datetime(2026, 7, 24, tzinfo=timezone.utc),
        )

    assert component.metadata == {"ecosystem": "pypi"}


def test_cached_dictionary_outputs_receive_snapshot_enrichment(tmp_path):
    path = _catalog(tmp_path)
    outputs = {
        "source": {
            "_v2": True,
            "components": [
                {
                    "name": "langchain",
                    "component_type": "dependency",
                    "metadata": {"ecosystem": "pypi"},
                }
            ],
            "relationships": [],
        }
    }

    enrich_analysis_outputs(
        outputs,
        db_path=path,
        no_network=True,
    )

    metadata = outputs["source"]["components"][0]["metadata"]
    assert metadata["liveness_status"] == "maintained"
    assert metadata["liveness_snapshot_at"] == "2026-07-01T00:00:00Z"


@pytest.mark.parametrize(
    "component",
    [
        AIComponent(
            name="langchain",
            component_type=AIComponentType.MODEL,
            metadata={"ecosystem": "pypi"},
        ),
        AIComponent(
            name="langchain",
            component_type=AIComponentType.DEPENDENCY,
            metadata={},
        ),
    ],
)
def test_non_package_components_are_not_queried(tmp_path, component):
    path = _catalog(tmp_path)

    with respx.mock:
        enrich_components(
            [component],
            db_path=path,
            freshness_url=FRESHNESS_URL,
            now=datetime(2026, 7, 24, tzinfo=timezone.utc),
        )

    assert "liveness_status" not in component.metadata


def test_manifest_advertises_catalog_and_freshness_endpoint(
    monkeypatch,
    tmp_path,
):
    catalog = tmp_path / "candidate.duckdb"
    catalog.touch()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "freshness_api": FRESHNESS_URL,
                "duckdb": {"filename": catalog.name},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AIBOM_MANIFEST_PATH", str(manifest))
    monkeypatch.delenv("AIBOM_DB_PATH", raising=False)
    monkeypatch.delenv("CISCO_AIBOM_FRESHNESS_URL", raising=False)

    assert resolve_package_catalog_path() == catalog
    assert resolve_freshness_url() == FRESHNESS_URL


def test_freshness_url_environment_override_wins(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"freshness_api": "https://manifest.example.test"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("AIBOM_MANIFEST_PATH", str(manifest))
    monkeypatch.setenv(
        "CISCO_AIBOM_FRESHNESS_URL",
        "https://override.example.test",
    )

    assert resolve_freshness_url() == "https://override.example.test"


def test_analyze_registers_snapshot_controls():
    root_command = get_command(app)
    analyze_command = root_command.commands["analyze"]
    option_names = {
        option
        for parameter in analyze_command.params
        for option in parameter.opts
    }

    assert "--no-network" in option_names
    assert "--liveness-only-snapshot" in option_names
