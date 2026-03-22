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

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path  # noqa: F401
from typing import Any
from urllib.parse import quote, urlparse

import httpx

__all__ = [
    "BitbucketAdapter",
    "GitHubAdapter",
    "GitLabAdapter",
    "RepoDiscovery",
    "RepoInfo",
    "get_adapter",
]

logger = logging.getLogger(__name__)

_PAGE_SIZE = 100
_MAX_PAGES = 10
_MAX_REPOS = _PAGE_SIZE * _MAX_PAGES
_CLIENT_TIMEOUT = 30.0


@dataclass
class RepoInfo:
    """Metadata about a discovered repository."""

    name: str
    full_name: str
    clone_url: str
    default_branch: str = "main"
    topics: list[str] = field(default_factory=list)
    language: str = ""
    description: str = ""
    is_archived: bool = False


class RepoDiscovery(ABC):
    """Interface for discovering repositories from a source code platform."""

    @abstractmethod
    def list_repos(
        self,
        namespace: str,
        *,
        name_filter: str | None = None,
        topic_filter: str | None = None,
        include_archived: bool = False,
    ) -> list[RepoInfo]:
        """List repositories in a namespace (org/group/project)."""
        ...


def _passes_filters(
    repo: RepoInfo,
    *,
    name_filter: str | None,
    topic_filter: str | None,
    include_archived: bool,
) -> bool:
    if not include_archived and repo.is_archived:
        return False
    if name_filter and name_filter.lower() not in repo.name.lower():
        return False
    if topic_filter:
        tlow = topic_filter.lower()
        if not any(t.lower() == tlow for t in repo.topics):
            return False
    return True


def _github_parse_next_url(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        chunk = part.strip()
        if 'rel="next"' not in chunk and "rel='next'" not in chunk:
            continue
        lt = chunk.find("<")
        gt = chunk.find(">", lt + 1)
        if lt == -1 or gt == -1:
            continue
        return chunk[lt + 1:gt]
    return None


class GitHubAdapter(RepoDiscovery):
    """Discover repos from GitHub organizations or users."""

    BASE = "https://api.github.com"

    def __init__(self, token: str | None = None, base_url: str | None = None):
        self._token = token
        self._base = (base_url or self.BASE).rstrip("/")
        self._headers = {"Accept": "application/vnd.github+json"}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"

    def _repo_from_json(self, r: dict[str, Any]) -> RepoInfo:
        topics = r.get("topics") or []
        if not isinstance(topics, list):
            topics = []
        lang = r.get("language") or ""
        if not isinstance(lang, str):
            lang = str(lang)
        desc = r.get("description") or ""
        if not isinstance(desc, str):
            desc = str(desc)
        branch = r.get("default_branch") or "main"
        if not isinstance(branch, str):
            branch = str(branch)
        return RepoInfo(
            name=str(r.get("name") or ""),
            full_name=str(r.get("full_name") or r.get("name") or ""),
            clone_url=str(r.get("clone_url") or ""),
            default_branch=branch,
            topics=[str(t) for t in topics],
            language=lang,
            description=desc,
            is_archived=bool(r.get("archived")),
        )

    def list_repos(
        self,
        namespace: str,
        *,
        name_filter: str | None = None,
        topic_filter: str | None = None,
        include_archived: bool = False,
    ) -> list[RepoInfo]:
        if not namespace.strip():
            return []
        ns = namespace.strip()
        q = quote(ns, safe="")
        org_ep = f"{self._base}/orgs/{q}/repos"
        user_ep = f"{self._base}/users/{q}/repos"
        out: list[RepoInfo] = []
        try:
            client_ctx = httpx.Client(
                timeout=_CLIENT_TIMEOUT,
                headers=self._headers,
            )
            with client_ctx as client:
                link_hdr: str | None = None
                for page in range(_MAX_PAGES):
                    if len(out) >= _MAX_REPOS:
                        break
                    try:
                        if page == 0:
                            r = client.get(
                                org_ep, params={"per_page": _PAGE_SIZE}
                            )
                            if r.status_code == 404:
                                r = client.get(
                                    user_ep, params={"per_page": _PAGE_SIZE}
                                )
                            elif r.status_code == 403:
                                logger.warning(
                                    "GitHub API returned %s for org %r: %s",
                                    r.status_code,
                                    ns,
                                    r.text[:500],
                                )
                                return []
                            if r.status_code in (404, 403):
                                logger.warning(
                                    "GitHub API returned %s for "
                                    "namespace %r: %s",
                                    r.status_code,
                                    ns,
                                    r.text[:500],
                                )
                                return []
                        else:
                            nxt = _github_parse_next_url(link_hdr)
                            if not nxt:
                                break
                            r = client.get(nxt)
                            if r.status_code in (404, 403):
                                logger.warning(
                                    "GitHub API returned %s while "
                                    "listing %r: %s",
                                    r.status_code,
                                    ns,
                                    r.text[:500],
                                )
                                break
                        r.raise_for_status()
                        batch = r.json()
                        if not isinstance(batch, list):
                            logger.warning(
                                "GitHub API returned non-list "
                                "payload for %r",
                                ns,
                            )
                            break
                        for item in batch:
                            if len(out) >= _MAX_REPOS:
                                break
                            if not isinstance(item, dict):
                                continue
                            repo = self._repo_from_json(item)
                            if _passes_filters(
                                repo,
                                name_filter=name_filter,
                                topic_filter=topic_filter,
                                include_archived=include_archived,
                            ):
                                out.append(repo)
                        link_hdr = r.headers.get("link")
                    except httpx.HTTPError as exc:
                        logger.warning(
                            "GitHub API request failed for %r: %s", ns, exc
                        )
                        break
        except httpx.HTTPError as exc:
            logger.warning("GitHub API client error for %r: %s", ns, exc)
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning(
                "GitHub API response parse error for %r: %s",
                ns,
                exc,
            )
        return out


class GitLabAdapter(RepoDiscovery):
    """Discover repos from GitLab groups."""

    BASE = "https://gitlab.com"

    def __init__(self, token: str | None = None, base_url: str | None = None):
        self._token = token
        root = (base_url or self.BASE).rstrip("/")
        parsed = urlparse(root)
        if parsed.path.rstrip("/") == "/api/v4":
            self._api_base = root.rstrip("/")
            self._origin = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
        else:
            self._origin = root
            self._api_base = f"{self._origin}/api/v4"
        self._headers: dict[str, str] = {}
        if token:
            self._headers["PRIVATE-TOKEN"] = token

    def _repo_from_json(self, p: dict[str, Any]) -> RepoInfo:
        topics = p.get("topics") or []
        if not isinstance(topics, list):
            topics = []
        lang = p.get("language") or ""
        if not isinstance(lang, str):
            lang = str(lang)
        desc = p.get("description") or ""
        if not isinstance(desc, str):
            desc = str(desc)
        branch = p.get("default_branch") or "main"
        if not isinstance(branch, str):
            branch = str(branch)
        path_ns = str(p.get("path_with_namespace") or p.get("path") or "")
        name = str(p.get("name") or "")
        return RepoInfo(
            name=name,
            full_name=path_ns or name,
            clone_url=str(p.get("http_url_to_repo") or ""),
            default_branch=branch,
            topics=[str(t) for t in topics],
            language=lang,
            description=desc,
            is_archived=bool(p.get("archived")),
        )

    def list_repos(
        self,
        namespace: str,
        *,
        name_filter: str | None = None,
        topic_filter: str | None = None,
        include_archived: bool = False,
    ) -> list[RepoInfo]:
        if not namespace.strip():
            return []
        ns = namespace.strip()
        enc = quote(ns, safe="")
        out: list[RepoInfo] = []
        try:
            client_ctx = httpx.Client(
                timeout=_CLIENT_TIMEOUT,
                headers=self._headers,
            )
            with client_ctx as client:
                group_url = (
                    f"{self._api_base}/groups/{enc}/projects"
                    f"?per_page={_PAGE_SIZE}&include_subgroups=true"
                )
                resp = client.get(group_url)
                if resp.status_code == 404:
                    is_group = False
                    user_url = (
                        f"{self._api_base}/users/{enc}/projects"
                        f"?per_page={_PAGE_SIZE}"
                    )
                    resp = client.get(user_url)
                    list_url = f"{self._api_base}/users/{enc}/projects"
                else:
                    is_group = True
                    list_url = f"{self._api_base}/groups/{enc}/projects"

                if resp.status_code in (404, 403):
                    logger.warning(
                        "GitLab API returned %s for namespace %r: %s",
                        resp.status_code,
                        ns,
                        resp.text[:500],
                    )
                    return []

                page = 1
                for _ in range(_MAX_PAGES):
                    if len(out) >= _MAX_REPOS:
                        break
                    try:
                        if page == 1:
                            r = resp
                        else:
                            params: dict[str, str | int] = {
                                "page": page,
                                "per_page": _PAGE_SIZE,
                            }
                            if is_group:
                                params["include_subgroups"] = "true"
                            r = client.get(list_url, params=params)
                        if r.status_code in (404, 403):
                            logger.warning(
                                "GitLab API returned %s while listing %r: %s",
                                r.status_code,
                                ns,
                                r.text[:500],
                            )
                            break
                        r.raise_for_status()
                        batch = r.json()
                        if not isinstance(batch, list):
                            logger.warning(
                                "GitLab API returned non-list "
                                "payload for %r",
                                ns,
                            )
                            break
                        for item in batch:
                            if len(out) >= _MAX_REPOS:
                                break
                            if not isinstance(item, dict):
                                continue
                            repo = self._repo_from_json(item)
                            if _passes_filters(
                                repo,
                                name_filter=name_filter,
                                topic_filter=topic_filter,
                                include_archived=include_archived,
                            ):
                                out.append(repo)
                        next_page = r.headers.get("x-next-page", "").strip()
                        if not next_page:
                            break
                        page = int(next_page)
                    except httpx.HTTPError as exc:
                        logger.warning(
                            "GitLab API request failed for %r: %s", ns, exc
                        )
                        break
                    except ValueError:
                        logger.warning(
                            "GitLab API invalid x-next-page for %r", ns
                        )
                        break
        except httpx.HTTPError as exc:
            logger.warning("GitLab API client error for %r: %s", ns, exc)
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning(
                "GitLab API response parse error for %r: %s",
                ns,
                exc,
            )
        return out


def _bitbucket_https_clone(links: Any) -> str:
    if not isinstance(links, dict):
        return ""
    clone = links.get("clone")
    if not isinstance(clone, list):
        return ""
    https = ""
    for entry in clone:
        if not isinstance(entry, dict):
            continue
        href = entry.get("href")
        name = entry.get("name")
        if not isinstance(href, str):
            continue
        if name == "https":
            return href
        if href.startswith("https://"):
            https = https or href
    return https


class BitbucketAdapter(RepoDiscovery):
    """Discover repos from Bitbucket Cloud workspaces/projects."""

    BASE = "https://api.bitbucket.org/2.0"

    def __init__(self, token: str | None = None):
        self._token = token
        self._headers: dict[str, str] = {}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"

    def _repo_from_json(self, r: dict[str, Any]) -> RepoInfo:
        links = r.get("links")
        clone_url = _bitbucket_https_clone(links)
        lang = r.get("language") or ""
        if not isinstance(lang, str):
            lang = str(lang)
        desc = r.get("description") or ""
        if not isinstance(desc, str):
            desc = str(desc)
        name = str(r.get("name") or "")
        full_name = str(r.get("full_name") or name)
        main = r.get("mainbranch")
        branch = "main"
        if isinstance(main, dict):
            bn = main.get("name")
            if isinstance(bn, str) and bn:
                branch = bn
        return RepoInfo(
            name=name,
            full_name=full_name,
            clone_url=clone_url,
            default_branch=branch,
            topics=[],
            language=lang,
            description=desc,
            is_archived=bool(r.get("archived")),
        )

    def list_repos(
        self,
        namespace: str,
        *,
        name_filter: str | None = None,
        topic_filter: str | None = None,
        include_archived: bool = False,
    ) -> list[RepoInfo]:
        if not namespace.strip():
            return []
        ws = namespace.strip()
        enc = quote(ws, safe="")
        out: list[RepoInfo] = []
        try:
            client_ctx = httpx.Client(
                timeout=_CLIENT_TIMEOUT,
                headers=self._headers,
            )
            with client_ctx as client:
                url: str | None = (
                    f"{self.BASE.rstrip('/')}/repositories/{enc}"
                    f"?pagelen={_PAGE_SIZE}"
                )
                page_count = 0
                while (
                    url
                    and page_count < _MAX_PAGES
                    and len(out) < _MAX_REPOS
                ):
                    try:
                        resp = client.get(url)
                        if resp.status_code in (404, 403):
                            logger.warning(
                                "Bitbucket API returned %s for "
                                "workspace %r: %s",
                                resp.status_code,
                                ws,
                                resp.text[:500],
                            )
                            if page_count == 0:
                                return out
                            break
                        resp.raise_for_status()
                        data = resp.json()
                        if not isinstance(data, dict):
                            logger.warning(
                                "Bitbucket API returned non-object "
                                "payload for %r",
                                ws,
                            )
                            break
                        values = data.get("values")
                        if not isinstance(values, list):
                            logger.warning(
                                "Bitbucket API missing values list "
                                "for %r",
                                ws,
                            )
                            break
                        for item in values:
                            if len(out) >= _MAX_REPOS:
                                break
                            if not isinstance(item, dict):
                                continue
                            repo = self._repo_from_json(item)
                            if _passes_filters(
                                repo,
                                name_filter=name_filter,
                                topic_filter=topic_filter,
                                include_archived=include_archived,
                            ):
                                out.append(repo)
                        page_count += 1
                        nxt = data.get("next")
                        url = nxt if isinstance(nxt, str) else None
                    except httpx.HTTPError as exc:
                        logger.warning(
                            "Bitbucket API request failed for %r: %s", ws, exc
                        )
                        break
        except httpx.HTTPError as exc:
            logger.warning("Bitbucket API client error for %r: %s", ws, exc)
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning(
                "Bitbucket API response parse error for %r: %s", ws, exc
            )
        return out


def get_adapter(
    platform: str,
    token: str | None = None,
    base_url: str | None = None,
) -> RepoDiscovery:
    """Return an adapter for the given platform name."""
    p = platform.lower()
    if p in ("github", "gh"):
        return GitHubAdapter(token=token, base_url=base_url)
    if p in ("gitlab", "gl"):
        return GitLabAdapter(token=token, base_url=base_url)
    if p in ("bitbucket", "bb"):
        return BitbucketAdapter(token=token)
    raise ValueError(f"Unknown platform: {platform!r}")
