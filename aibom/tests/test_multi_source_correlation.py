# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for §18 multi-source (repo + image) correlation.

Covers the container-aware cross-source link builders: image-baked ENV
producers, shared base image, SBOM-backed shared dependencies, and the
derived-from-repo advisory.
"""

from __future__ import annotations

from pathlib import Path

from aibom.cross_repo_links import build_deterministic_cross_repo_links
from aibom.models.enums import CrossRepoLinkType


def _consumer_component(env_var: str, file_path: str) -> dict:
    return {
        "name": "endpoint",
        "instance_id": "c1",
        "file_path": file_path,
        "line_number": 1,
        "component_type": "llm_endpoint",
        "model_name": None,
        "metadata": {"env": env_var},
    }


class TestImageBakedEnvProducer:
    def test_image_env_binds_to_consumer_repo(self, tmp_path: Path) -> None:
        img = tmp_path / "img"
        api = tmp_path / "api"
        img.mkdir()
        api.mkdir()

        per_repo_results = {
            str(img): {"components": [], "relationships": []},
            str(api): {
                "components": [
                    _consumer_component("MODEL_ENDPOINT_URL", str(api / "app.py"))
                ],
                "relationships": [],
            },
        }
        source_metadata = {
            str(img): {
                "kind": "container",
                "source_name": "svc:latest",
                "image_env": {"MODEL_ENDPOINT_URL": "https://llm.internal"},
                "base_image": "",
                "sbom_packages": [],
                "source_repo_url": "",
            },
        }

        links = build_deterministic_cross_repo_links(
            per_repo_results, [str(img), str(api)], source_metadata
        )
        env_links = [
            link for link in links if link.link_type == CrossRepoLinkType.ENV_VAR_BINDING
        ]
        assert any(link.identifier == "MODEL_ENDPOINT_URL" for link in env_links)
        binding = next(link for link in env_links if link.identifier == "MODEL_ENDPOINT_URL")
        roles = {o.role for o in binding.occurrences}
        assert "producer" in roles
        assert "consumer" in roles

    def test_no_binding_without_image_env(self, tmp_path: Path) -> None:
        img = tmp_path / "img"
        api = tmp_path / "api"
        img.mkdir()
        api.mkdir()
        per_repo_results = {
            str(img): {"components": [], "relationships": []},
            str(api): {
                "components": [
                    _consumer_component("MODEL_ENDPOINT_URL", str(api / "app.py"))
                ],
                "relationships": [],
            },
        }
        links = build_deterministic_cross_repo_links(
            per_repo_results, [str(img), str(api)], source_metadata={}
        )
        assert not [
            link for link in links if link.link_type == CrossRepoLinkType.ENV_VAR_BINDING
        ]


class TestSharedBaseImage:
    def test_two_images_sharing_base(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        per_repo_results = {
            str(a): {"components": [], "relationships": []},
            str(b): {"components": [], "relationships": []},
        }
        source_metadata = {
            str(a): {"kind": "container", "source_name": "a:1",
                     "base_image": "python:3.12-slim", "sbom_packages": [],
                     "image_env": {}, "source_repo_url": ""},
            str(b): {"kind": "container", "source_name": "b:1",
                     "base_image": "python:3.12-slim", "sbom_packages": [],
                     "image_env": {}, "source_repo_url": ""},
        }
        links = build_deterministic_cross_repo_links(
            per_repo_results, [str(a), str(b)], source_metadata
        )
        base_links = [
            link for link in links if link.link_type == CrossRepoLinkType.SHARED_BASE_IMAGE
        ]
        assert len(base_links) == 1
        assert base_links[0].identifier == "python:3.12-slim"
        assert len(base_links[0].occurrences) == 2

    def test_distinct_base_no_link(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        per_repo_results = {
            str(a): {"components": [], "relationships": []},
            str(b): {"components": [], "relationships": []},
        }
        source_metadata = {
            str(a): {"kind": "container", "source_name": "a:1",
                     "base_image": "python:3.12-slim", "sbom_packages": [],
                     "image_env": {}, "source_repo_url": ""},
            str(b): {"kind": "container", "source_name": "b:1",
                     "base_image": "node:20", "sbom_packages": [],
                     "image_env": {}, "source_repo_url": ""},
        }
        links = build_deterministic_cross_repo_links(
            per_repo_results, [str(a), str(b)], source_metadata
        )
        assert not [
            link for link in links if link.link_type == CrossRepoLinkType.SHARED_BASE_IMAGE
        ]


class TestSbomSharedDependency:
    def test_shared_sbom_package(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        per_repo_results = {
            str(a): {"components": [], "relationships": []},
            str(b): {"components": [], "relationships": []},
        }
        source_metadata = {
            str(a): {"kind": "container", "source_name": "a:1", "base_image": "",
                     "sbom_packages": ["pkg:deb/debian/openssl@3.0",
                                       "pkg:pypi/torch@2.1"],
                     "image_env": {}, "source_repo_url": ""},
            str(b): {"kind": "container", "source_name": "b:1", "base_image": "",
                     "sbom_packages": ["pkg:deb/debian/openssl@3.0"],
                     "image_env": {}, "source_repo_url": ""},
        }
        links = build_deterministic_cross_repo_links(
            per_repo_results, [str(a), str(b)], source_metadata
        )
        sbom_deps = [
            link for link in links
            if link.link_type == CrossRepoLinkType.SHARED_DEPENDENCY
            and link.evidence_type == "sbom"
        ]
        assert len(sbom_deps) == 1
        assert sbom_deps[0].identifier == "pkg:deb/debian/openssl@3.0"


class TestDerivedFromRepo:
    def test_advisory_when_upstream_unscanned(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        per_repo_results = {
            str(a): {"components": [], "relationships": []},
            str(b): {"components": [], "relationships": []},
        }
        source_metadata = {
            str(a): {"kind": "container", "source_name": "a:1", "base_image": "",
                     "sbom_packages": [], "image_env": {},
                     "source_repo_url": "https://github.com/org/upstream"},
            str(b): {"kind": "container", "source_name": "b:1", "base_image": "",
                     "sbom_packages": [], "image_env": {}, "source_repo_url": ""},
        }
        links = build_deterministic_cross_repo_links(
            per_repo_results, [str(a), str(b)], source_metadata
        )
        derived = [
            link for link in links if link.link_type == CrossRepoLinkType.DERIVED_FROM_REPO
        ]
        assert len(derived) == 1
        assert derived[0].evidence_type == "unscanned_upstream_repo"
        assert derived[0].identifier == "https://github.com/org/upstream"

    def test_no_advisory_when_upstream_scanned(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        per_repo_results = {
            str(a): {"components": [], "relationships": []},
            str(b): {"components": [], "relationships": []},
        }
        # The upstream repo IS scanned (source b is that git URL), so no advisory.
        source_metadata = {
            str(a): {"kind": "container", "source_name": "a:1", "base_image": "",
                     "sbom_packages": [], "image_env": {},
                     "source_repo_url": "https://github.com/org/upstream.git"},
            str(b): {"kind": "git-url",
                     "source_name": "https://github.com/org/upstream",
                     "base_image": "", "sbom_packages": [], "image_env": {},
                     "source_repo_url": ""},
        }
        links = build_deterministic_cross_repo_links(
            per_repo_results, [str(a), str(b)], source_metadata
        )
        assert not [
            link for link in links if link.link_type == CrossRepoLinkType.DERIVED_FROM_REPO
        ]

    def test_no_advisory_when_upstream_scanned_as_local_checkout(
        self, tmp_path: Path
    ) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        per_repo_results = {
            str(a): {"components": [], "relationships": []},
            str(b): {"components": [], "relationships": []},
        }
        # Upstream is scanned as a LOCAL checkout (kind=local-path) whose resolved
        # git remote (repo_url, SSH spelling) matches the image's HTTPS source
        # label — canonicalization must treat them as the same repo, so no
        # advisory fires.
        source_metadata = {
            str(a): {"kind": "container", "source_name": "a:1", "base_image": "",
                     "sbom_packages": [], "image_env": {},
                     "source_repo_url": "https://github.com/org/upstream"},
            str(b): {"kind": "local-path", "source_name": str(b),
                     "base_image": "", "sbom_packages": [], "image_env": {},
                     "source_repo_url": "",
                     "repo_url": "git@github.com:org/upstream.git"},
        }
        links = build_deterministic_cross_repo_links(
            per_repo_results, [str(a), str(b)], source_metadata
        )
        assert not [
            link for link in links if link.link_type == CrossRepoLinkType.DERIVED_FROM_REPO
        ]


class TestBackwardCompatibility:
    def test_omitting_source_metadata_is_safe(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        per_repo_results = {
            str(a): {"components": [], "relationships": []},
            str(b): {"components": [], "relationships": []},
        }
        # Old two-arg call site must still work.
        links = build_deterministic_cross_repo_links(
            per_repo_results, [str(a), str(b)]
        )
        assert links == []
