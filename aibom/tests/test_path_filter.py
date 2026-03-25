# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aibom.utils.path_filter import SKIP_DIR_NAMES, should_skip_dir


class TestSkipDirNames:
    def test_exact_git(self):
        assert should_skip_dir(".git")

    def test_exact_venv(self):
        assert should_skip_dir("venv")
        assert should_skip_dir(".venv")

    def test_exact_node_modules(self):
        assert should_skip_dir("node_modules")

    def test_exact_pycache(self):
        assert should_skip_dir("__pycache__")

    def test_exact_site_packages(self):
        assert should_skip_dir("site-packages")

    def test_exact_tox(self):
        assert should_skip_dir(".tox")

    def test_exact_build_dist(self):
        assert should_skip_dir("build")
        assert should_skip_dir("dist")


class TestVendoredVenvPatterns:
    def test_underscore_venv_suffix(self):
        assert should_skip_dir("orchestrator-tes_venv")
        assert should_skip_dir("myapp_venv")

    def test_hyphen_venv_suffix(self):
        assert should_skip_dir("project-venv")
        assert should_skip_dir("svc-venv")

    def test_egg_info_suffix(self):
        assert should_skip_dir("mypackage.egg-info")
        assert should_skip_dir("aibom-0.5.1.egg-info")

    def test_case_insensitive_venv(self):
        assert should_skip_dir("APP_VENV")
        assert should_skip_dir("Service_Venv")
        assert should_skip_dir("MY-VENV")


class TestNonSkipped:
    def test_regular_directory(self):
        assert not should_skip_dir("src")
        assert not should_skip_dir("lib")
        assert not should_skip_dir("app")

    def test_partial_match_not_skipped(self):
        assert not should_skip_dir("my_venv_backup")
        assert not should_skip_dir("venv_old")

    def test_empty_string(self):
        assert not should_skip_dir("")

    def test_venv_as_substring(self):
        assert not should_skip_dir("venvs")
        assert not should_skip_dir("virtualenv")


class TestConstantCompleteness:
    def test_skip_dir_names_is_frozenset(self):
        assert isinstance(SKIP_DIR_NAMES, frozenset)

    def test_expected_entries_present(self):
        expected = {".git", "__pycache__", "node_modules", ".venv", "venv",
                    ".tox", "site-packages", "dist", "build"}
        assert expected.issubset(SKIP_DIR_NAMES)
