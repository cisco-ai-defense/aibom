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

"""Deterministic cross-repo link builder.

Generates :class:`CrossRepoLink` objects WITHOUT the LLM by matching
env-var bindings, shared model IDs, and shared dependencies across
per-repo scan results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from .cross_ref import CrossRefIndex, EnvVarEntry, build_env_index, build_package_index
from .models.enums import AIComponentType, CrossRepoLinkType
from .models.scan import CrossRepoLink, RepoOccurrence
from .scanners.remote_agent_resolver import _normalize_url_for_index
from .source_attribution import canonicalize_git_remote

_LOGGER = logging.getLogger(__name__)


def _scanned_repo_identities(
    source_metadata: dict[str, dict[str, Any]],
) -> set[str]:
    """Canonical git identities of every scanned source.

    Uses each source's ``repo_url`` (the git remote the CLI resolved — set for
    git-URL sources AND local checkouts of a git repo), falling back to the
    ``source_name`` for git-URL sources when ``repo_url`` is absent. Canonicalized
    via :func:`canonicalize_git_remote` so SSH/HTTPS spellings and ``.git``
    suffixes compare equal. A local checkout therefore counts as "scanned" and
    suppresses a spurious derived-from-repo advisory for an image built from it.
    """
    scanned: set[str] = set()
    for meta in source_metadata.values():
        url = (meta.get("repo_url") or "").strip()
        if not url and meta.get("kind") == "git-url":
            url = (meta.get("source_name") or "").strip()
        if not url:
            continue
        canon = canonicalize_git_remote(url)
        if canon:
            scanned.add(canon)
    return scanned


def _repo_for_file(file_path: str, scan_paths: list[str]) -> str | None:
    """Return the scan-path root that contains *file_path*."""
    try:
        fp = Path(file_path).resolve()
    except OSError:
        return None
    for sp in scan_paths:
        try:
            root = Path(sp).resolve()
            fp.relative_to(root)
            return sp
        except (OSError, ValueError):
            continue
    return None


def _build_env_var_binding_links(
    per_repo_results: dict[str, dict[str, Any]],
    env_index: CrossRefIndex,
    scan_paths: list[str],
) -> list[CrossRepoLink]:
    """Match env vars defined in one repo and referenced in another."""
    links: list[CrossRepoLink] = []

    env_consumers: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for repo_path, data in per_repo_results.items():
        for c in data.get("components", []):
            meta = c.metadata if hasattr(c, "metadata") else c.get("metadata", {})
            env_name = meta.get("env") or meta.get("env_var")
            if not env_name:
                continue
            env_consumers.setdefault(env_name, []).append((repo_path, c))

    for env_name, consumers in env_consumers.items():
        entries = env_index.env.get(env_name, [])
        if not entries:
            continue

        producer_repos: set[str] = set()
        producer_occurrences: list[RepoOccurrence] = []
        for entry in entries:
            repo = _repo_for_file(entry.source_path, scan_paths)
            if repo and repo not in producer_repos:
                producer_repos.add(repo)
                producer_occurrences.append(RepoOccurrence(
                    repo_path=repo,
                    file_path=entry.source_path,
                    role="producer",
                ))

        consumer_repos: set[str] = set()
        consumer_occurrences: list[RepoOccurrence] = []
        for repo_path, comp in consumers:
            if repo_path not in consumer_repos:
                consumer_repos.add(repo_path)
                comp_name = comp.name if hasattr(comp, "name") else comp.get("name", "")
                comp_iid = comp.instance_id if hasattr(comp, "instance_id") else comp.get("instance_id", "")
                comp_fp = comp.file_path if hasattr(comp, "file_path") else comp.get("file_path", "")
                comp_ln = comp.line_number if hasattr(comp, "line_number") else comp.get("line_number", 0)
                consumer_occurrences.append(RepoOccurrence(
                    repo_path=repo_path,
                    component_name=comp_name,
                    component_instance_id=comp_iid,
                    file_path=comp_fp,
                    line_number=comp_ln,
                    role="consumer",
                ))

        all_repos = producer_repos | consumer_repos
        if len(all_repos) < 2:
            continue

        resolved = entries[0].value if entries else ""
        links.append(CrossRepoLink(
            link_type=CrossRepoLinkType.ENV_VAR_BINDING,
            identifier=env_name,
            resolved_value=resolved,
            occurrences=producer_occurrences + consumer_occurrences,
            evidence=f"Env var '{env_name}' defined in {len(producer_repos)} repo(s), consumed in {len(consumer_repos)} repo(s)",
        ))

    return links


def _build_shared_model_links(
    per_repo_results: dict[str, dict[str, Any]],
) -> list[CrossRepoLink]:
    """Match identical model_name values across repos."""
    model_occurrences: dict[str, list[RepoOccurrence]] = {}

    for repo_path, data in per_repo_results.items():
        seen_models: set[str] = set()
        for c in data.get("components", []):
            mn = c.model_name if hasattr(c, "model_name") else c.get("model_name")
            if not mn or mn in seen_models:
                continue
            ct = c.component_type if hasattr(c, "component_type") else c.get("component_type")
            ct_val = ct.value if hasattr(ct, "value") else str(ct)
            if ct_val not in ("model", "embedding", "llm_endpoint", "model_endpoint"):
                continue
            seen_models.add(mn)
            comp_name = c.name if hasattr(c, "name") else c.get("name", "")
            comp_iid = c.instance_id if hasattr(c, "instance_id") else c.get("instance_id", "")
            comp_fp = c.file_path if hasattr(c, "file_path") else c.get("file_path", "")
            comp_ln = c.line_number if hasattr(c, "line_number") else c.get("line_number", 0)
            model_occurrences.setdefault(mn, []).append(RepoOccurrence(
                repo_path=repo_path,
                component_name=comp_name,
                component_instance_id=comp_iid,
                file_path=comp_fp,
                line_number=comp_ln,
                role="shared",
            ))

    links: list[CrossRepoLink] = []
    for model_name, occs in model_occurrences.items():
        repos = {o.repo_path for o in occs}
        if len(repos) < 2:
            continue
        links.append(CrossRepoLink(
            link_type=CrossRepoLinkType.SHARED_MODEL,
            identifier=model_name,
            resolved_value=model_name,
            occurrences=occs,
            evidence=f"Model '{model_name}' used in {len(repos)} repos",
        ))

    return links


def _build_shared_dependency_links(
    scan_paths: list[str],
) -> list[CrossRepoLink]:
    """Match packages appearing in multiple repos' manifests."""
    pkg_repos: dict[str, set[str]] = {}
    for root in scan_paths:
        sub_index: CrossRefIndex = build_package_index([root])
        for pkg in sub_index.packages:
            pkg_repos.setdefault(pkg, set()).add(root)

    links: list[CrossRepoLink] = []
    for pkg, repos in sorted(pkg_repos.items()):
        if len(repos) < 2:
            continue
        links.append(CrossRepoLink(
            link_type=CrossRepoLinkType.SHARED_DEPENDENCY,
            identifier=pkg,
            occurrences=[
                RepoOccurrence(repo_path=repo, role="shared")
                for repo in sorted(repos)
            ],
            evidence=f"Package '{pkg}' found in {len(repos)} repos",
        ))

    return links


def _propagate_cross_repo_endpoint_values(
    links: list[CrossRepoLink],
    per_repo_results: dict[str, dict[str, Any]],
) -> None:
    """Back-propagate resolved env-var URLs into consumer components.

    When an ``ENV_VAR_BINDING`` link has a ``resolved_value`` (the actual
    URL from the producer repo), update matching consumer components that
    still have ``model_name=None``.
    """
    for link in links:
        if link.link_type != CrossRepoLinkType.ENV_VAR_BINDING:
            continue
        if not link.resolved_value:
            continue
        env_name = link.identifier

        for repo_path, data in per_repo_results.items():
            comps = data.get("components", [])
            for i, c in enumerate(comps):
                meta = c.metadata if hasattr(c, "metadata") else c.get("metadata", {})
                c_env = meta.get("env") or meta.get("env_var") or ""
                if c_env != env_name:
                    continue
                c_mn = c.model_name if hasattr(c, "model_name") else c.get("model_name")
                if c_mn:
                    continue
                ct = c.component_type if hasattr(c, "component_type") else c.get("component_type")
                ct_val = ct.value if hasattr(ct, "value") else str(ct)
                if ct_val not in ("llm_endpoint", "model_endpoint", "vector_store"):
                    continue

                new_meta = dict(meta)
                new_meta["endpoint_url"] = link.resolved_value
                new_meta["env_var"] = env_name
                new_meta["cross_repo_resolved"] = True
                if hasattr(c, "model_copy"):
                    updates: dict[str, Any] = {
                        "model_name": None,
                        "heuristic_confidence": max(c.heuristic_confidence, 0.7),
                        "metadata": new_meta,
                    }
                    c_name = c.name if hasattr(c, "name") else c.get("name", "")
                    if c_name.startswith("env:") or not c_name.startswith("http"):
                        updates["name"] = link.resolved_value
                    comps[i] = c.model_copy(update=updates)
                _LOGGER.info(
                    "Cross-repo propagation: %s → model_name='%s' (from %s)",
                    env_name, link.resolved_value, repo_path,
                )


def _build_shared_endpoint_links(
    per_repo_results: dict[str, dict[str, Any]],
) -> list[CrossRepoLink]:
    """Match identical endpoint URLs across repos.

    If the same endpoint URL appears as a component in two different repos
    (e.g., code reads the URL in one repo, Helm deploys it in another),
    create a cross-repo link.
    """
    _ENDPOINT_TYPES = frozenset({
        "llm_endpoint", "model_endpoint", "vector_store",
    })

    url_occurrences: dict[str, list[RepoOccurrence]] = {}
    for repo_path, data in per_repo_results.items():
        seen_urls: set[str] = set()
        for c in data.get("components", []):
            ct = c.component_type if hasattr(c, "component_type") else c.get("component_type")
            ct_val = ct.value if hasattr(ct, "value") else str(ct)
            if ct_val not in _ENDPOINT_TYPES:
                continue
            name = c.name if hasattr(c, "name") else c.get("name", "")
            if not name or not name.startswith(("http://", "https://")):
                continue
            url_norm = name.rstrip("/")
            if url_norm in seen_urls:
                continue
            seen_urls.add(url_norm)
            meta = c.metadata if hasattr(c, "metadata") else c.get("metadata", {})
            comp_fp = c.file_path if hasattr(c, "file_path") else c.get("file_path", "")
            comp_ln = c.line_number if hasattr(c, "line_number") else c.get("line_number", 0)
            env_var = meta.get("env_var", "")
            url_occurrences.setdefault(url_norm, []).append(RepoOccurrence(
                repo_path=repo_path,
                component_name=name,
                file_path=comp_fp,
                line_number=comp_ln,
                role=f"endpoint (env: {env_var})" if env_var else "endpoint",
            ))

    links: list[CrossRepoLink] = []
    for url, occs in url_occurrences.items():
        repos = {o.repo_path for o in occs}
        if len(repos) < 2:
            continue
        links.append(CrossRepoLink(
            link_type=CrossRepoLinkType.ENV_VAR_BINDING,
            identifier=url,
            resolved_value=url,
            occurrences=occs,
            evidence=f"Endpoint URL '{url[:60]}…' referenced in {len(repos)} repos",
        ))

    return links


_AGENT_RELATIONSHIP_TYPES = frozenset({
    "USES_MODEL", "USES_TOOL", "USES_VECTOR_STORE", "USES_KNOWLEDGE_BASE",
    "USES_EMBEDDING", "USES_LLM_ENDPOINT", "USES_MCP_SERVER",
})

_AGENT_COMPONENT_TYPES = frozenset({
    "agent", "mcp_client", "mcp_server",
})


def _build_agent_model_cross_links(
    shared_model_links: list[CrossRepoLink],
    per_repo_results: dict[str, dict[str, Any]],
) -> list[CrossRepoLink]:
    """Enrich SHARED_MODEL links with agent context from intra-repo relationships.

    For each shared model, check if any repo has an intra-repo relationship
    where an agent/component uses that model. If the agent is in repo A and
    the model is deployed in repo B, emit a cross-repo link with explicit
    source (agent) and target (model) roles.
    """
    links: list[CrossRepoLink] = []

    model_to_repos: dict[str, set[str]] = {}
    for link in shared_model_links:
        for occ in link.occurrences:
            model_to_repos.setdefault(link.identifier, set()).add(occ.repo_path)

    agent_uses: dict[str, list[tuple[str, str, str]]] = {}
    for repo_path, data in per_repo_results.items():
        for rel in data.get("relationships", []):
            rt = rel.relationship_type if hasattr(rel, "relationship_type") else rel.get("relationship_type", "")
            rt_val = rt.value if hasattr(rt, "value") else str(rt)
            if rt_val not in _AGENT_RELATIONSHIP_TYPES:
                continue
            st = rel.source_type if hasattr(rel, "source_type") else rel.get("source_type", "")
            st_val = st.value if hasattr(st, "value") else str(st)
            if st_val not in _AGENT_COMPONENT_TYPES:
                continue
            tn = rel.target_name if hasattr(rel, "target_name") else rel.get("target_name", "")
            sn = rel.source_name if hasattr(rel, "source_name") else rel.get("source_name", "")
            if tn:
                agent_uses.setdefault(tn, []).append((repo_path, sn, rt_val))

    for model_name, repos in model_to_repos.items():
        usages = agent_uses.get(model_name, [])
        for agent_repo, agent_name, rel_type in usages:
            other_repos = repos - {agent_repo}
            for target_repo in other_repos:
                links.append(CrossRepoLink(
                    link_type=CrossRepoLinkType.CUSTOM,
                    identifier=f"{agent_name} → {model_name}",
                    resolved_value=model_name,
                    occurrences=[
                        RepoOccurrence(
                            repo_path=agent_repo,
                            component_name=agent_name,
                            role="source",
                        ),
                        RepoOccurrence(
                            repo_path=target_repo,
                            component_name=model_name,
                            role="target",
                        ),
                    ],
                    evidence=(
                        f"{agent_name} ({agent_repo.split('/')[-1]}) "
                        f"--[{rel_type}]--> {model_name} "
                        f"(deployed in {target_repo.split('/')[-1]})"
                    ),
                ))

    return links


def _getattr_or_key(obj: Any, attr: str, default: Any = None) -> Any:
    """Fetch *attr* from Pydantic models (attr) or dicts (key).

    Several cross-repo helpers in this module already duplicate this
    polymorphism inline. ``_build_a2a_agent_cross_links`` uses this
    helper to keep the A2A wiring readable while preserving the
    per_repo_results format the rest of the module already accepts.
    """
    if hasattr(obj, attr):
        return getattr(obj, attr)
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return default


def _build_a2a_agent_cross_links(
    per_repo_results: dict[str, dict[str, Any]],
) -> list[CrossRepoLink]:
    """Link ``AGENT_PROXY`` clients to ``AGENT`` cards across repos.

    For every :data:`~aibom.models.enums.AIComponentType.AGENT_PROXY`
    whose local resolution ended at ``unverified_url_pattern`` (i.e.
    the target Agent Card was not in the same repo), search every
    other scanned repo for an :data:`~aibom.models.enums.
    AIComponentType.AGENT` component whose
    ``metadata['agent_card']['endpoints']`` contains a matching URL.
    On a cross-repo hit:

    1. Emit an :data:`~aibom.models.enums.CrossRepoLinkType.
       A2A_AGENT_CLIENT_SERVER` link with the proxy as ``source`` and
       the card as ``target``.
    2. Upgrade the proxy's ``metadata['remote_verification']`` in
       place to ``verified_cross_repo_card`` (confidence 0.9), so
       downstream consumers see the stronger match without a second
       scan.

    Proxies that were already locally matched
    (``verified_local_card``) are not considered — their resolution
    is authoritative for their own repo.
    """
    card_index: dict[str, list[tuple[str, Any]]] = {}
    for repo_path, data in per_repo_results.items():
        for c in data.get("components", []):
            ct = _getattr_or_key(c, "component_type")
            ct_val = ct.value if hasattr(ct, "value") else str(ct)
            if ct_val != AIComponentType.AGENT.value:
                continue
            fw = _getattr_or_key(c, "framework", "") or ""
            if fw != "a2a":
                continue
            meta = _getattr_or_key(c, "metadata", {}) or {}
            card = meta.get("agent_card") or {}
            endpoints = card.get("endpoints") or []
            for raw in endpoints:
                if not isinstance(raw, str):
                    continue
                canon = _normalize_url_for_index(raw)
                if not canon:
                    continue
                card_index.setdefault(canon, []).append((repo_path, c))
                try:
                    parsed = urlparse(canon)
                except ValueError:
                    continue
                host_root = urlunparse(
                    (parsed.scheme, parsed.netloc, "", "", "", "")
                )
                if host_root and host_root != canon:
                    card_index.setdefault(host_root, []).append(
                        (repo_path, c)
                    )

    links: list[CrossRepoLink] = []

    for proxy_repo, data in per_repo_results.items():
        comps = data.get("components", [])
        for idx, proxy in enumerate(comps):
            ct = _getattr_or_key(proxy, "component_type")
            ct_val = ct.value if hasattr(ct, "value") else str(ct)
            if ct_val != AIComponentType.AGENT_PROXY.value:
                continue
            proxy_meta = _getattr_or_key(proxy, "metadata", {}) or {}
            verification = proxy_meta.get("remote_verification") or {}
            if verification.get("status") != "unverified_url_pattern":
                continue

            raw_url = proxy_meta.get("remote_url") or ""
            canon = _normalize_url_for_index(raw_url)
            if not canon:
                continue
            matches = card_index.get(canon)
            if not matches:
                try:
                    parsed = urlparse(canon)
                except ValueError:
                    continue
                host_root = urlunparse(
                    (parsed.scheme, parsed.netloc, "", "", "", "")
                )
                matches = card_index.get(host_root)
            if not matches:
                continue

            target_repo: str | None = None
            target_card: Any = None
            for card_repo, card in matches:
                if card_repo != proxy_repo:
                    target_repo = card_repo
                    target_card = card
                    break
            if target_repo is None or target_card is None:
                continue

            proxy_name = _getattr_or_key(proxy, "name", "") or ""
            proxy_iid = _getattr_or_key(proxy, "instance_id", "") or ""
            proxy_fp = _getattr_or_key(proxy, "file_path", "") or ""
            proxy_ln = _getattr_or_key(proxy, "line_number", 0) or 0
            card_name = _getattr_or_key(target_card, "name", "") or ""
            card_iid = _getattr_or_key(target_card, "instance_id", "") or ""
            card_fp = _getattr_or_key(target_card, "file_path", "") or ""
            card_ln = _getattr_or_key(target_card, "line_number", 0) or 0

            links.append(CrossRepoLink(
                link_type=CrossRepoLinkType.A2A_AGENT_CLIENT_SERVER,
                identifier=canon,
                resolved_value=card_name,
                occurrences=[
                    RepoOccurrence(
                        repo_path=proxy_repo,
                        component_name=proxy_name,
                        component_instance_id=proxy_iid,
                        file_path=proxy_fp,
                        line_number=proxy_ln,
                        role="source",
                    ),
                    RepoOccurrence(
                        repo_path=target_repo,
                        component_name=card_name,
                        component_instance_id=card_iid,
                        file_path=card_fp,
                        line_number=card_ln,
                        role="target",
                    ),
                ],
                evidence=(
                    f"A2A proxy '{proxy_name}' in "
                    f"{Path(proxy_repo).name} invokes remote agent "
                    f"'{card_name}' in {Path(target_repo).name} "
                    f"via {canon}"
                ),
            ))

            if hasattr(proxy, "model_copy"):
                new_meta = dict(proxy_meta)
                new_meta["remote_verification"] = {
                    "status": "verified_cross_repo_card",
                    "confidence": 0.9,
                    "matched_component_instance_id": card_iid,
                    "matched_component_name": card_name,
                    "match_source": "cross_repo",
                    "matched_repo_path": target_repo,
                }
                comps[idx] = proxy.model_copy(
                    update={
                        "metadata": new_meta,
                        "heuristic_confidence": max(
                            _getattr_or_key(
                                proxy, "heuristic_confidence", 0.0
                            ) or 0.0,
                            0.9,
                        ),
                    }
                )

    return links


def _augment_env_index_with_image_env(
    env_index: CrossRefIndex,
    source_metadata: dict[str, dict[str, Any]],
) -> int:
    """Fold image-baked ``ENV`` vars into *env_index* as producers.

    A container image whose config bakes ``ENV FOO=bar`` acts as a
    producer of ``FOO`` for cross-source ``ENV_VAR_BINDING``. The entry's
    ``source_path`` is set to the image's scan-path root so
    :func:`_repo_for_file` attributes it to that source. Returns the
    number of env vars added.
    """
    added = 0
    for scan_path, meta in source_metadata.items():
        image_env = meta.get("image_env") or {}
        if not isinstance(image_env, dict):
            continue
        for name, value in image_env.items():
            env_index.env.setdefault(str(name), []).append(EnvVarEntry(
                name=str(name),
                value="" if value is None else str(value),
                source_type="image-env",
                source_path=scan_path,
            ))
            added += 1
    return added


def _build_shared_base_image_links(
    source_metadata: dict[str, dict[str, Any]],
) -> list[CrossRepoLink]:
    """Link two or more image sources that share the same base image."""
    base_to_paths: dict[str, list[str]] = {}
    for scan_path, meta in source_metadata.items():
        base = (meta.get("base_image") or "").strip()
        if not base:
            continue
        base_to_paths.setdefault(base, []).append(scan_path)

    links: list[CrossRepoLink] = []
    for base, paths in sorted(base_to_paths.items()):
        if len(set(paths)) < 2:
            continue
        links.append(CrossRepoLink(
            link_type=CrossRepoLinkType.SHARED_BASE_IMAGE,
            identifier=base,
            resolved_value=base,
            occurrences=[
                RepoOccurrence(repo_path=p, role="shares base image")
                for p in sorted(set(paths))
            ],
            evidence=f"Base image '{base}' shared by {len(set(paths))} image sources",
        ))
    return links


def _build_sbom_shared_dependency_links(
    source_metadata: dict[str, dict[str, Any]],
) -> list[CrossRepoLink]:
    """SBOM-backed ``SHARED_DEPENDENCY`` links from Syft package sets.

    Distinct from manifest-based shared deps: identifiers are package
    URLs (purls) and links carry ``evidence_type='sbom'``.
    """
    pkg_to_paths: dict[str, set[str]] = {}
    for scan_path, meta in source_metadata.items():
        for pkg in meta.get("sbom_packages") or []:
            pkg_to_paths.setdefault(str(pkg), set()).add(scan_path)

    links: list[CrossRepoLink] = []
    for pkg, paths in sorted(pkg_to_paths.items()):
        if len(paths) < 2:
            continue
        links.append(CrossRepoLink(
            link_type=CrossRepoLinkType.SHARED_DEPENDENCY,
            identifier=pkg,
            evidence_type="sbom",
            occurrences=[
                RepoOccurrence(repo_path=p, role="shared")
                for p in sorted(paths)
            ],
            evidence=f"Package '{pkg}' present in {len(paths)} image SBOMs",
        ))
    return links


def _build_derived_from_repo_links(
    source_metadata: dict[str, dict[str, Any]],
) -> list[CrossRepoLink]:
    """Advisory ``DERIVED_FROM_REPO`` links from image source labels.

    When an image declares ``org.opencontainers.image.source`` and that
    upstream repo was **not** among the scanned sources, emit an advisory
    link (``evidence_type='unscanned_upstream_repo'``) so the operator
    knows the image's provenance repo is unscanned.
    """
    scanned_urls = _scanned_repo_identities(source_metadata)

    links: list[CrossRepoLink] = []
    for scan_path, meta in source_metadata.items():
        url = (meta.get("source_repo_url") or "").strip()
        if not url:
            continue
        # A local checkout of this same repo (kind=local-path) counts as scanned
        # via its resolved git remote, so canonicalize both sides before the
        # membership test to avoid a false "unscanned upstream" advisory.
        if canonicalize_git_remote(url) in scanned_urls:
            continue
        links.append(CrossRepoLink(
            link_type=CrossRepoLinkType.DERIVED_FROM_REPO,
            identifier=url,
            resolved_value=url,
            evidence_type="unscanned_upstream_repo",
            occurrences=[
                RepoOccurrence(repo_path=scan_path, role="derived image"),
                RepoOccurrence(repo_path=url, role="unscanned upstream"),
            ],
            evidence=(
                f"Image derived from upstream repo '{url}' "
                f"(not included in this scan)"
            ),
        ))
    return links


def build_deterministic_cross_repo_links(
    per_repo_results: dict[str, dict[str, Any]],
    scan_paths: list[str],
    source_metadata: dict[str, dict[str, Any]] | None = None,
) -> list[CrossRepoLink]:
    """Build cross-repo links deterministically (no LLM).

    Detects:
    - ``ENV_VAR_BINDING``: env vars defined in one repo, consumed in another
      (image-baked ``ENV`` vars participate as producers when
      *source_metadata* supplies them).
    - ``SHARED_MODEL``: identical model IDs across repos.
    - ``SHARED_DEPENDENCY``: packages in multiple repos' manifests, plus
      SBOM-backed matches (``evidence_type='sbom'``) across image sources.
    - ``SHARED_BASE_IMAGE``: image sources sharing an OCI base image.
    - ``DERIVED_FROM_REPO``: advisory for an image whose upstream source
      repo was not scanned.
    - ``SHARED_ENDPOINT``: identical endpoint URLs across repos.
    - ``A2A_AGENT_CLIENT_SERVER``: an ``AGENT_PROXY`` in one repo whose
      URL matches an ``AGENT`` card's A2A endpoint in another repo.

    *source_metadata* (optional) maps each scan-path to per-source image
    metadata (``kind``, ``source_name``, ``base_image``, ``sbom_packages``,
    ``image_env``, ``source_repo_url``) and enables the container-aware
    correlations. ``scan_paths`` must be **real on-disk paths** — image
    refs / git URLs will not resolve and are warned about.

    Also propagates resolved env-var values to consumer components and
    upgrades unverified A2A proxies to ``verified_cross_repo_card``.
    """
    if len(per_repo_results) < 2:
        return []

    for sp in scan_paths:
        if not Path(sp).exists():
            _LOGGER.warning(
                "Cross-source correlation: scan path not found on disk, "
                "env/dependency indexing will skip it: %s", sp,
            )

    source_metadata = source_metadata or {}

    env_index = build_env_index(scan_paths)
    n_image_env = _augment_env_index_with_image_env(env_index, source_metadata)
    if n_image_env:
        _LOGGER.info(
            "Cross-source correlation: folded %d image-baked ENV var(s) into "
            "the env index as producers", n_image_env,
        )

    links: list[CrossRepoLink] = []
    env_links = _build_env_var_binding_links(per_repo_results, env_index, scan_paths)
    links.extend(env_links)
    _propagate_cross_repo_endpoint_values(env_links, per_repo_results)
    shared_model_links = _build_shared_model_links(per_repo_results)
    links.extend(shared_model_links)
    links.extend(_build_agent_model_cross_links(shared_model_links, per_repo_results))
    links.extend(_build_shared_endpoint_links(per_repo_results))
    links.extend(_build_shared_dependency_links(scan_paths))
    links.extend(_build_sbom_shared_dependency_links(source_metadata))
    links.extend(_build_shared_base_image_links(source_metadata))
    links.extend(_build_derived_from_repo_links(source_metadata))
    links.extend(_build_a2a_agent_cross_links(per_repo_results))

    links = _resolve_colon_prefixed_repo_paths(links, scan_paths)
    links = _filter_intra_repo_links(links)
    links = _filter_quality_bar(links)
    n_env = sum(1 for link in links if link.link_type == CrossRepoLinkType.ENV_VAR_BINDING)
    n_model = sum(1 for link in links if link.link_type == CrossRepoLinkType.SHARED_MODEL)
    n_dep = sum(1 for link in links if link.link_type == CrossRepoLinkType.SHARED_DEPENDENCY)
    n_base = sum(1 for link in links if link.link_type == CrossRepoLinkType.SHARED_BASE_IMAGE)
    n_derived = sum(
        1 for link in links if link.link_type == CrossRepoLinkType.DERIVED_FROM_REPO
    )
    _LOGGER.info(
        "Deterministic cross-repo links: %d total (%d env-var/endpoint, "
        "%d model, %d dep, %d base-image, %d derived-from-repo)",
        len(links), n_env, n_model, n_dep, n_base, n_derived,
    )
    return links


def _resolve_colon_prefixed_repo_paths(
    links: list[CrossRepoLink],
    scan_paths: list[str],
) -> list[CrossRepoLink]:
    """Resolve colon-prefixed names (e.g. ``:repo-name:``) to actual scan paths."""
    short_to_path: dict[str, str] = {}
    for sp in scan_paths:
        short_to_path[Path(sp).name] = sp

    result: list[CrossRepoLink] = []
    for link in links:
        new_occs: list[RepoOccurrence] = []
        for occ in link.occurrences:
            if occ.repo_path and not occ.repo_path.startswith(":"):
                new_occs.append(occ)
            elif occ.repo_path:
                name = occ.repo_path.strip(":")
                resolved = short_to_path.get(name)
                if resolved:
                    new_occs.append(occ.model_copy(update={"repo_path": resolved}))
                else:
                    new_occs.append(occ)
            else:
                new_occs.append(occ)
        result.append(link.model_copy(update={"occurrences": new_occs}))
    return result


def _filter_intra_repo_links(links: list[CrossRepoLink]) -> list[CrossRepoLink]:
    """Remove links where all occurrences resolve to the same repo."""
    result: list[CrossRepoLink] = []
    for link in links:
        repos = {o.repo_path for o in link.occurrences if o.repo_path}
        if len(repos) >= 2:
            result.append(link)
        else:
            _LOGGER.debug("Filtered intra-repo link: %s", link.identifier)
    return result


def _filter_quality_bar(links: list[CrossRepoLink]) -> list[CrossRepoLink]:
    """Discard links where all occurrences lack ``repo_path``."""
    result: list[CrossRepoLink] = []
    for link in links:
        if any(o.repo_path for o in link.occurrences):
            result.append(link)
        else:
            _LOGGER.debug("Filtered no-repo-path link: %s", link.identifier)
    return result
