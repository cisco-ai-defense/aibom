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

"""Opportunistically refresh package liveness already present in an AI BOM."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, MutableMapping, cast

import httpx

from .db_loader import _load_manifest
from .package_registry import (
    PackageCoordinate,
    PackageSnapshot,
    package_coordinate,
    read_package_snapshots,
    resolve_package_catalog_path,
)

LOGGER = logging.getLogger(__name__)

FRESHNESS_MAX_AGE = timedelta(days=7)
FRESHNESS_BATCH_SIZE = 100
FRESHNESS_TIMEOUT = httpx.Timeout(5.0)


def resolve_freshness_url() -> str | None:
    """Resolve an explicitly configured or manifest-advertised endpoint."""

    configured = os.environ.get("CISCO_AIBOM_FRESHNESS_URL")
    if configured:
        return configured
    manifest, _ = _load_manifest()
    if not manifest:
        return None
    value = manifest.get("freshness_api")
    return value if isinstance(value, str) and value else None


def enrich_analysis_outputs(
    outputs: MutableMapping[str, Any],
    *,
    db_path: Path | str | None = None,
    freshness_url: str | None = None,
    no_network: bool = False,
    liveness_only_snapshot: bool = False,
    now: datetime | None = None,
    client: httpx.Client | None = None,
) -> None:
    """Enrich dependencies without changing deterministic scan caches."""

    components: list[Any] = []
    for output in outputs.values():
        if not isinstance(output, dict) or not output.get("_v2"):
            continue
        value = output.get("components")
        if isinstance(value, list):
            components.extend(value)
    enrich_components(
        components,
        db_path=db_path,
        freshness_url=freshness_url,
        no_network=no_network,
        liveness_only_snapshot=liveness_only_snapshot,
        now=now,
        client=client,
    )


def enrich_components(
    components: Iterable[Any],
    *,
    db_path: Path | str | None = None,
    freshness_url: str | None = None,
    no_network: bool = False,
    liveness_only_snapshot: bool = False,
    now: datetime | None = None,
    client: httpx.Client | None = None,
) -> None:
    """Attach snapshot data and, when needed, anonymous live deltas."""

    components_by_coordinate: dict[
        PackageCoordinate,
        list[Any],
    ] = defaultdict(list)
    for component in components:
        coordinate = _component_coordinate(component)
        if coordinate is not None:
            components_by_coordinate[coordinate].append(component)
    if not components_by_coordinate:
        return

    selected_path = (
        db_path if db_path is not None else resolve_package_catalog_path()
    )
    snapshots = read_package_snapshots(
        selected_path,
        components_by_coordinate.keys(),
    )
    if not snapshots:
        return

    for coordinate, snapshot in snapshots.items():
        for component in components_by_coordinate[coordinate]:
            _apply_snapshot(component, snapshot)

    if no_network or liveness_only_snapshot:
        return
    endpoint = (
        freshness_url
        if freshness_url is not None
        else resolve_freshness_url()
    )
    if not endpoint:
        return

    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    else:
        current_time = current_time.astimezone(timezone.utc)
    stale_by_snapshot: dict[str, list[PackageCoordinate]] = defaultdict(list)
    for coordinate, snapshot in snapshots.items():
        parsed_snapshot_at = _parse_timestamp(snapshot.liveness_snapshot_at)
        if parsed_snapshot_at is None:
            continue
        if current_time - parsed_snapshot_at > FRESHNESS_MAX_AGE:
            snapshot_key = snapshot.liveness_snapshot_at or ""
            stale_by_snapshot[snapshot_key].append(coordinate)

    owns_client = client is None
    http_client = client or httpx.Client(timeout=FRESHNESS_TIMEOUT)
    try:
        for snapshot_key, coordinates in stale_by_snapshot.items():
            for batch in _batches(coordinates, FRESHNESS_BATCH_SIZE):
                _apply_live_batch(
                    http_client,
                    endpoint,
                    snapshot_key,
                    batch,
                    components_by_coordinate,
                )
    finally:
        if owns_client:
            http_client.close()


def _component_coordinate(component: Any) -> PackageCoordinate | None:
    component_type = _component_value(component, "component_type")
    if hasattr(component_type, "value"):
        component_type = component_type.value
    if component_type != "dependency":
        return None
    metadata = _component_value(component, "metadata")
    if not isinstance(metadata, dict):
        return None
    ecosystem = metadata.get("ecosystem")
    name = _component_value(component, "name")
    if not isinstance(ecosystem, str) or not isinstance(name, str):
        return None
    if not ecosystem.strip() or not name.strip():
        return None
    return package_coordinate(ecosystem, name)


def _component_value(component: Any, key: str) -> Any:
    if isinstance(component, dict):
        return component.get(key)
    return getattr(component, key, None)


def _metadata(component: Any) -> dict[str, Any]:
    if isinstance(component, dict):
        value = component.setdefault("metadata", {})
        if not isinstance(value, dict):
            value = {}
            component["metadata"] = value
    else:
        value = component.metadata
    return cast(dict[str, Any], value)


def _apply_snapshot(component: Any, snapshot: PackageSnapshot) -> None:
    metadata = _metadata(component)
    if snapshot.liveness_status is not None:
        metadata["liveness_status"] = snapshot.liveness_status
    if snapshot.liveness_snapshot_at is not None:
        metadata["liveness_snapshot_at"] = snapshot.liveness_snapshot_at
    if snapshot.certification is not None:
        metadata["liveness_certification"] = snapshot.certification


def _apply_live_batch(
    client: httpx.Client,
    endpoint: str,
    snapshot_at: str,
    coordinates: list[PackageCoordinate],
    components_by_coordinate: dict[PackageCoordinate, list[Any]],
) -> None:
    requested_coordinates = set(coordinates)
    payload = {
        "snapshot_at": snapshot_at,
        "packages": [
            {"ecosystem": item.ecosystem, "name": item.name}
            for item in coordinates
        ],
    }
    try:
        response = client.post(endpoint, json=payload)
        if response.status_code == 429:
            LOGGER.warning(
                "Package freshness request was rate limited; "
                "using snapshot data."
            )
            return
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        LOGGER.warning(
            "Package freshness request failed; using snapshot data: %s",
            exc,
        )
        return
    if not isinstance(body, dict):
        return
    as_of = _parse_timestamp(body.get("as_of"))
    snapshot_time = _parse_timestamp(snapshot_at)
    if as_of is None or snapshot_time is None or as_of <= snapshot_time:
        return
    signals = body.get("signals")
    if not isinstance(signals, list):
        return
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        ecosystem = signal.get("ecosystem")
        name = signal.get("package_name")
        if not isinstance(ecosystem, str) or not isinstance(name, str):
            continue
        coordinate = package_coordinate(ecosystem, name)
        if coordinate not in requested_coordinates:
            continue
        for component in components_by_coordinate.get(coordinate, []):
            metadata = _metadata(component)
            status = signal.get("liveness_status")
            if isinstance(status, str) and status:
                metadata["liveness_status"] = status
            metadata["as_of"] = _isoformat(as_of)
            observed_at = _parse_timestamp(signal.get("observed_at"))
            if observed_at is not None:
                metadata["liveness_observed_at"] = _isoformat(observed_at)
            certification = signal.get("certification")
            if isinstance(certification, dict):
                metadata["liveness_certification"] = certification


def _batches(
    items: list[PackageCoordinate],
    size: int,
) -> Iterable[list[PackageCoordinate]]:
    for index in range(0, len(items), size):
        yield items[index:index + size]


def _parse_timestamp(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            normalized = value.strip().replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
