# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import httpx
import respx
import pytest

from aibom.platform_adapters import (
    BitbucketAdapter,
    GitHubAdapter,
    GitLabAdapter,
    RepoInfo,
    get_adapter,
)


def _gh_repo(
    name: str,
    *,
    archived: bool = False,
    topics: list[str] | None = None,
) -> dict:
    return {
        "name": name,
        "full_name": f"org/{name}",
        "clone_url": f"https://github.com/org/{name}.git",
        "default_branch": "main",
        "topics": topics or [],
        "language": "Python",
        "description": f"{name} repo",
        "archived": archived,
    }


def _gl_repo(name: str, group: str = "my-group") -> dict:
    return {
        "name": name,
        "path_with_namespace": f"{group}/{name}",
        "http_url_to_repo": f"https://gitlab.com/{group}/{name}.git",
        "default_branch": "main",
        "topics": [],
        "description": f"{name} project",
    }


_GH_ORG_REPOS = "https://api.github.com/orgs/{ns}/repos?per_page=100"
_GH_USER_REPOS = "https://api.github.com/users/{ns}/repos?per_page=100"
_GL_GROUP_PROJECTS = (
    "https://gitlab.com/api/v4/groups/{ns}/projects"
    "?per_page=100&include_subgroups=true"
)
_GL_GROUP_PROJECTS_INTERNAL = (
    "https://gitlab.internal.com/api/v4/groups/{ns}/projects"
    "?per_page=100&include_subgroups=true"
)
_BB_WS = "https://api.bitbucket.org/2.0/repositories/{ws}?pagelen=100"


class TestGetAdapter:
    def test_github_aliases(self) -> None:
        assert isinstance(get_adapter("github"), GitHubAdapter)
        assert isinstance(get_adapter("gh"), GitHubAdapter)

    def test_gitlab_aliases(self) -> None:
        assert isinstance(get_adapter("gitlab"), GitLabAdapter)
        assert isinstance(get_adapter("gl"), GitLabAdapter)

    def test_bitbucket_aliases(self) -> None:
        assert isinstance(get_adapter("bitbucket"), BitbucketAdapter)
        assert isinstance(get_adapter("bb"), BitbucketAdapter)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            get_adapter("unknown")


class TestGitHubAdapter:
    @respx.mock(assert_all_called=False)
    def test_list_org_repos(self, respx_mock: respx.MockRouter) -> None:
        payload = json.loads(json.dumps([_gh_repo("r1"), _gh_repo("r2")]))
        respx_mock.get(_GH_ORG_REPOS.format(ns="my-org")).mock(
            return_value=httpx.Response(200, json=payload)
        )
        repos = GitHubAdapter().list_repos("my-org")
        assert len(repos) == 2
        assert repos[0] == RepoInfo(
            name="r1",
            full_name="org/r1",
            clone_url="https://github.com/org/r1.git",
            default_branch="main",
            topics=[],
            language="Python",
            description="r1 repo",
            is_archived=False,
        )
        assert repos[1].name == "r2"

    @respx.mock(assert_all_called=False)
    def test_falls_back_to_user(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get(_GH_ORG_REPOS.format(ns="my-user")).mock(
            return_value=httpx.Response(404)
        )
        respx_mock.get(_GH_USER_REPOS.format(ns="my-user")).mock(
            return_value=httpx.Response(200, json=[_gh_repo("solo")])
        )
        repos = GitHubAdapter().list_repos("my-user")
        assert len(repos) == 1
        assert repos[0].name == "solo"

    @respx.mock(assert_all_called=False)
    def test_name_filter(self, respx_mock: respx.MockRouter) -> None:
        payload = [
            _gh_repo("frontend"),
            _gh_repo("api-gateway"),
            _gh_repo("billing-api"),
        ]
        respx_mock.get(_GH_ORG_REPOS.format(ns="o")).mock(
            return_value=httpx.Response(200, json=payload)
        )
        repos = GitHubAdapter().list_repos("o", name_filter="api")
        assert [r.name for r in repos] == ["api-gateway", "billing-api"]

    @respx.mock(assert_all_called=False)
    def test_topic_filter(self, respx_mock: respx.MockRouter) -> None:
        payload = [
            _gh_repo("a", topics=["ai", "ml"]),
            _gh_repo("b", topics=[]),
            _gh_repo("c", topics=["other"]),
        ]
        respx_mock.get(_GH_ORG_REPOS.format(ns="o")).mock(
            return_value=httpx.Response(200, json=payload)
        )
        repos = GitHubAdapter().list_repos("o", topic_filter="ai")
        assert len(repos) == 1
        assert repos[0].name == "a"

    @respx.mock(assert_all_called=False)
    def test_excludes_archived(self, respx_mock: respx.MockRouter) -> None:
        payload = [_gh_repo("live"), _gh_repo("old", archived=True)]
        respx_mock.get(_GH_ORG_REPOS.format(ns="o")).mock(
            return_value=httpx.Response(200, json=payload)
        )
        repos = GitHubAdapter().list_repos("o")
        assert [r.name for r in repos] == ["live"]

    @respx.mock(assert_all_called=False)
    def test_includes_archived(self, respx_mock: respx.MockRouter) -> None:
        payload = [_gh_repo("live"), _gh_repo("old", archived=True)]
        respx_mock.get(_GH_ORG_REPOS.format(ns="o")).mock(
            return_value=httpx.Response(200, json=payload)
        )
        repos = GitHubAdapter().list_repos("o", include_archived=True)
        assert [r.name for r in repos] == ["live", "old"]

    @respx.mock(assert_all_called=False)
    def test_auth_header(self, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.get(_GH_ORG_REPOS.format(ns="my-org")).mock(
            return_value=httpx.Response(200, json=json.loads("[]"))
        )
        GitHubAdapter(token="ghp_xxx").list_repos("my-org")
        assert route.called
        hdrs = route.calls.last.request.headers
        assert hdrs["Authorization"] == "Bearer ghp_xxx"

    @respx.mock(assert_all_called=False)
    def test_api_error_returns_empty(
        self, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.get(_GH_ORG_REPOS.format(ns="my-org")).mock(
            return_value=httpx.Response(500)
        )
        assert GitHubAdapter().list_repos("my-org") == []


class TestGitLabAdapter:
    @respx.mock(assert_all_called=False)
    def test_list_group_repos(self, respx_mock: respx.MockRouter) -> None:
        payload = [_gl_repo("p1")]
        respx_mock.get(_GL_GROUP_PROJECTS.format(ns="my-group")).mock(
            return_value=httpx.Response(200, json=payload)
        )
        repos = GitLabAdapter().list_repos("my-group")
        assert len(repos) == 1
        assert repos[0] == RepoInfo(
            name="p1",
            full_name="my-group/p1",
            clone_url="https://gitlab.com/my-group/p1.git",
            default_branch="main",
            topics=[],
            language="",
            description="p1 project",
            is_archived=False,
        )

    @respx.mock(assert_all_called=False)
    def test_name_filter(self, respx_mock: respx.MockRouter) -> None:
        payload = [
            _gl_repo("frontend", group="g"),
            _gl_repo("api-gateway", group="g"),
            _gl_repo("billing-api", group="g"),
        ]
        respx_mock.get(_GL_GROUP_PROJECTS.format(ns="g")).mock(
            return_value=httpx.Response(200, json=payload)
        )
        repos = GitLabAdapter().list_repos("g", name_filter="api")
        assert [r.name for r in repos] == ["api-gateway", "billing-api"]

    @respx.mock(assert_all_called=False)
    def test_auth_header(self, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.get(_GL_GROUP_PROJECTS.format(ns="g")).mock(
            return_value=httpx.Response(200, json=json.loads("[]"))
        )
        GitLabAdapter(token="gl_xxx").list_repos("g")
        assert route.called
        assert route.calls.last.request.headers["PRIVATE-TOKEN"] == "gl_xxx"

    @respx.mock(assert_all_called=False)
    def test_custom_base_url(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get(_GL_GROUP_PROJECTS_INTERNAL.format(ns="my-group")).mock(
            return_value=httpx.Response(200, json=json.loads("[]"))
        )
        GitLabAdapter(base_url="https://gitlab.internal.com").list_repos(
            "my-group"
        )


class TestBitbucketAdapter:
    @respx.mock(assert_all_called=False)
    def test_list_workspace_repos(self, respx_mock: respx.MockRouter) -> None:
        body = {
            "values": [
                {
                    "name": "repo1",
                    "full_name": "my-workspace/repo1",
                    "links": {
                        "clone": [
                            {
                                "name": "https",
                                "href": (
                                    "https://bitbucket.org/"
                                    "my-workspace/repo1.git"
                                ),
                            }
                        ]
                    },
                    "mainbranch": {"name": "main"},
                    "language": "python",
                    "description": "desc",
                }
            ],
            "next": None,
        }
        respx_mock.get(_BB_WS.format(ws="my-workspace")).mock(
            return_value=httpx.Response(200, json=body)
        )
        repos = BitbucketAdapter().list_repos("my-workspace")
        assert len(repos) == 1
        assert repos[0] == RepoInfo(
            name="repo1",
            full_name="my-workspace/repo1",
            clone_url="https://bitbucket.org/my-workspace/repo1.git",
            default_branch="main",
            topics=[],
            language="python",
            description="desc",
            is_archived=False,
        )

    @respx.mock(assert_all_called=False)
    def test_pagination(self, respx_mock: respx.MockRouter) -> None:
        page2_url = (
            "https://api.bitbucket.org/2.0/repositories/w?page=2"
        )
        page1 = {
            "values": [
                {
                    "name": "r1",
                    "full_name": "w/r1",
                    "links": {
                        "clone": [
                            {
                                "name": "https",
                                "href": "https://bitbucket.org/w/r1.git",
                            }
                        ]
                    },
                    "mainbranch": {"name": "main"},
                    "language": None,
                    "description": None,
                }
            ],
            "next": page2_url,
        }
        page2 = {
            "values": [
                {
                    "name": "r2",
                    "full_name": "w/r2",
                    "links": {
                        "clone": [
                            {
                                "name": "https",
                                "href": "https://bitbucket.org/w/r2.git",
                            }
                        ]
                    },
                    "mainbranch": {"name": "develop"},
                    "language": None,
                    "description": None,
                }
            ],
            "next": None,
        }
        respx_mock.get(_BB_WS.format(ws="w")).mock(
            return_value=httpx.Response(200, json=page1)
        )
        respx_mock.get(page2_url).mock(
            return_value=httpx.Response(200, json=page2)
        )
        repos = BitbucketAdapter().list_repos("w")
        assert [r.name for r in repos] == ["r1", "r2"]


class TestRepoInfo:
    def test_defaults(self) -> None:
        r = RepoInfo(
            name="n",
            full_name="o/n",
            clone_url="https://example.com/o/n.git",
        )
        assert r.default_branch == "main"
        assert r.topics == []
        assert r.language == ""
        assert r.description == ""
        assert r.is_archived is False
