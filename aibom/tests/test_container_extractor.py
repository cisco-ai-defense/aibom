# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for the container source extraction module."""

import json
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aibom.scanners.container_extractor import (
    ExtractionResult,
    ImageConfig,
    VALID_TIERS,
    _discover_from_tarball,
    _entrypoint_target_dir,
    _extract_from_tarball,
    _find_runtime,
    _identify_app_dirs,
    _is_dependency_path,
    _is_system_path,
    _parse_image_config,
    extract_source_from_image,
    validate_tier,
)


class TestImageConfigParsing:
    def test_parses_standard_config(self):
        raw = {
            "config": {
                "WorkingDir": "/app",
                "Entrypoint": ["python", "main.py"],
                "Cmd": None,
                "Env": ["MODEL=gpt-4o", "PATH=/usr/bin"],
            }
        }
        config = _parse_image_config(raw)
        assert config.workdir == "/app"
        assert config.entrypoint == ["python", "main.py"]
        assert config.cmd == []
        assert config.env["MODEL"] == "gpt-4o"

    def test_handles_empty_config(self):
        config = _parse_image_config({})
        assert config.workdir == "/"
        assert config.entrypoint == []
        assert config.cmd == []

    def test_uppercase_config_key(self):
        raw = {"Config": {"WorkingDir": "/srv"}}
        config = _parse_image_config(raw)
        assert config.workdir == "/srv"

    def test_base64_encoded_config(self):
        import base64
        inner = json.dumps({
            "config": {"WorkingDir": "/opt/app", "Env": ["KEY=val"]},
        })
        b64 = base64.b64encode(inner.encode()).decode()
        config = _parse_image_config(b64)
        assert config.workdir == "/opt/app"
        assert config.env["KEY"] == "val"

    def test_json_string_config(self):
        raw_str = json.dumps({"config": {"WorkingDir": "/srv/api"}})
        config = _parse_image_config(raw_str)
        assert config.workdir == "/srv/api"

    def test_unparseable_string_returns_defaults(self):
        config = _parse_image_config("not-json-not-base64")
        assert config.workdir == "/"
        assert config.entrypoint == []

    def test_parses_oci_labels(self):
        raw = {
            "config": {
                "WorkingDir": "/app",
                "Labels": {
                    "org.opencontainers.image.base.name": "python:3.12-slim",
                    "org.opencontainers.image.source": "https://github.com/org/svc",
                    "maintainer": None,
                },
            }
        }
        config = _parse_image_config(raw)
        assert config.labels["org.opencontainers.image.base.name"] == "python:3.12-slim"
        assert config.labels["org.opencontainers.image.source"] == "https://github.com/org/svc"
        assert config.labels["maintainer"] == ""

    def test_missing_labels_default_empty(self):
        config = _parse_image_config({"config": {"WorkingDir": "/app"}})
        assert config.labels == {}


class TestSyftSbomParsing:
    def test_extracts_purls(self):
        from aibom.scanners.container_extractor import _sbom_packages_from_syft

        data = {
            "artifacts": [
                {"name": "openssl", "version": "3.0", "type": "deb",
                 "purl": "pkg:deb/debian/openssl@3.0"},
                {"name": "requests", "version": "2.31", "type": "python",
                 "purl": "pkg:pypi/requests@2.31"},
            ]
        }
        pkgs = _sbom_packages_from_syft(data)
        assert "pkg:deb/debian/openssl@3.0" in pkgs
        assert "pkg:pypi/requests@2.31" in pkgs

    def test_synthesizes_id_when_no_purl(self):
        from aibom.scanners.container_extractor import _sbom_packages_from_syft

        data = {"artifacts": [{"name": "libfoo", "version": "1.2", "type": "deb"}]}
        assert _sbom_packages_from_syft(data) == ["deb/libfoo@1.2"]

    def test_dedups_and_ignores_malformed(self):
        from aibom.scanners.container_extractor import _sbom_packages_from_syft

        data = {
            "artifacts": [
                {"purl": "pkg:pypi/x@1"},
                {"purl": "pkg:pypi/x@1"},
                "not-a-dict",
                {"version": "1"},
            ]
        }
        assert _sbom_packages_from_syft(data) == ["pkg:pypi/x@1"]

    def test_empty_when_no_artifacts(self):
        from aibom.scanners.container_extractor import _sbom_packages_from_syft

        assert _sbom_packages_from_syft({}) == []


class TestEntrypointTargetDir:
    def test_python_script(self):
        config = ImageConfig(entrypoint=["python", "/opt/myservice/main.py"])
        assert _entrypoint_target_dir(config) == "/opt/myservice"

    def test_node_script(self):
        config = ImageConfig(cmd=["node", "/srv/app/index.js"])
        assert _entrypoint_target_dir(config) == "/srv/app"

    def test_no_script(self):
        config = ImageConfig(entrypoint=["bash"])
        assert _entrypoint_target_dir(config) is None

    def test_system_path_excluded(self):
        config = ImageConfig(entrypoint=["python", "/usr/bin/gunicorn"])
        assert _entrypoint_target_dir(config) is None


class TestPathClassification:
    def test_system_paths(self):
        assert _is_system_path("/usr/bin/python")
        assert _is_system_path("/var/log/app.log")
        assert _is_system_path("/etc/nginx/nginx.conf")
        assert not _is_system_path("/app/main.py")
        assert not _is_system_path("/opt/service/run.py")

    def test_dependency_paths(self):
        assert _is_dependency_path("/usr/lib/python3.11/site-packages/torch")
        assert _is_dependency_path("/app/node_modules/express")
        assert not _is_dependency_path("/app/src/main.py")


class TestIdentifyAppDirs:
    def test_workdir_signal(self):
        config = ImageConfig(workdir="/app")
        dirs, agentic, _ = _identify_app_dirs(config, ["/app/main.py"])
        assert "/app" in dirs
        assert not agentic

    def test_entrypoint_signal(self):
        config = ImageConfig(
            workdir="/",
            entrypoint=["python", "/opt/service/app.py"],
        )
        dirs, _, _ = _identify_app_dirs(config, ["/opt/service/app.py"])
        assert "/opt/service" in dirs

    def test_source_file_signal(self):
        config = ImageConfig(workdir="/")
        files = ["/myapp/main.py", "/myapp/utils.py", "/usr/bin/python"]
        dirs, _, _ = _identify_app_dirs(config, files)
        assert "/myapp" in dirs

    def test_excludes_system_and_deps(self):
        config = ImageConfig(workdir="/")
        files = [
            "/usr/lib/python3.11/site-packages/torch/__init__.py",
            "/usr/bin/python",
        ]
        dirs, _, _ = _identify_app_dirs(config, files)
        assert "/usr" not in dirs

    def test_fallback_dirs(self):
        config = ImageConfig(workdir="/")
        files = ["/app/config.yaml"]
        dirs, _, _ = _identify_app_dirs(config, files)
        assert "/app" in dirs

    def test_agentic_for_ambiguous(self):
        config = ImageConfig(workdir="/")
        files = [
            f"/{d}/main.py"
            for d in ("serviceA", "serviceB", "serviceC", "serviceD", "config")
        ]
        dirs, agentic, hint = _identify_app_dirs(config, files)
        assert agentic
        assert "Ambiguous" in hint


class TestTarballDiscovery:
    def _build_docker_save_tar(self, tmp_dir: Path, config: dict, layer_files: dict[str, bytes]) -> Path:
        tar_path = tmp_dir / "image.tar"
        config_bytes = json.dumps(config).encode()
        manifest = json.dumps([{"Config": "config.json", "Layers": ["layer/layer.tar"]}]).encode()

        layer_tar_path = tmp_dir / "layer.tar"
        with tarfile.open(layer_tar_path, "w") as lt:
            for fname, content in layer_files.items():
                import io
                info = tarfile.TarInfo(name=fname)
                info.size = len(content)
                lt.addfile(info, io.BytesIO(content))

        with tarfile.open(tar_path, "w") as tf:
            import io
            ci = tarfile.TarInfo(name="config.json")
            ci.size = len(config_bytes)
            tf.addfile(ci, io.BytesIO(config_bytes))

            mi = tarfile.TarInfo(name="manifest.json")
            mi.size = len(manifest)
            tf.addfile(mi, io.BytesIO(manifest))

            tf.add(str(layer_tar_path), arcname="layer/layer.tar")

        return tar_path

    def test_discovers_config_and_files(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            tar_path = self._build_docker_save_tar(
                td_path,
                {"config": {"WorkingDir": "/app", "Env": ["MODEL=gpt-4"]}},
                {"app/main.py": b"import openai\n"},
            )
            config, files = _discover_from_tarball(tar_path)
            assert config.workdir == "/app"
            assert config.env.get("MODEL") == "gpt-4"
            assert any("main.py" in f for f in files)


class TestOCITarballDiscovery:
    """Tests for OCI-format tarballs (blobs/sha256/<hash>)."""

    def _build_oci_tar(self, tmp_dir: Path, config: dict, layer_files: dict[str, bytes]) -> Path:
        tar_path = tmp_dir / "oci_image.tar"
        config_bytes = json.dumps(config).encode()

        layer_tar_path = tmp_dir / "layer_inner.tar"
        import io
        with tarfile.open(layer_tar_path, "w") as lt:
            for fname, content in layer_files.items():
                info = tarfile.TarInfo(name=fname)
                info.size = len(content)
                lt.addfile(info, io.BytesIO(content))

        blob_name = "blobs/sha256/abc123layer"
        config_blob = "blobs/sha256/abc123config"

        manifest = json.dumps([{
            "Config": config_blob,
            "Layers": [blob_name],
        }]).encode()

        with tarfile.open(tar_path, "w") as tf:
            ci = tarfile.TarInfo(name=config_blob)
            ci.size = len(config_bytes)
            tf.addfile(ci, io.BytesIO(config_bytes))

            mi = tarfile.TarInfo(name="manifest.json")
            mi.size = len(manifest)
            tf.addfile(mi, io.BytesIO(manifest))

            tf.add(str(layer_tar_path), arcname=blob_name)

        return tar_path

    def test_oci_discovery(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            tar_path = self._build_oci_tar(
                td_path,
                {"config": {"WorkingDir": "/myapp", "Env": ["GPU=true"]}},
                {"myapp/serve.py": b"import flask\n"},
            )
            config, files = _discover_from_tarball(tar_path)
            assert config.workdir == "/myapp"
            assert config.env["GPU"] == "true"
            assert any("serve.py" in f for f in files)

    def test_oci_extraction(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            tar_path = self._build_oci_tar(
                td_path,
                {"config": {"WorkingDir": "/app"}},
                {"app/run.py": b"print('hello')\n", "etc/config": b"x=1\n"},
            )
            dest = td_path / "out"
            dest.mkdir()
            _extract_from_tarball(tar_path, ["/app"], dest)
            assert (dest / "app" / "run.py").exists()
            assert not (dest / "etc" / "config").exists()


class TestTarballExtraction:
    def test_extracts_matching_paths(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            layer_tar = td_path / "layer.tar"

            import io
            with tarfile.open(layer_tar, "w") as lt:
                content = b"import torch\n"
                info = tarfile.TarInfo(name="app/model.py")
                info.size = len(content)
                lt.addfile(info, io.BytesIO(content))

                sys_content = b"#!/bin/bash\n"
                sys_info = tarfile.TarInfo(name="usr/bin/script.sh")
                sys_info.size = len(sys_content)
                lt.addfile(sys_info, io.BytesIO(sys_content))

            image_tar = td_path / "image.tar"
            config_bytes = json.dumps({"config": {"WorkingDir": "/app"}}).encode()
            manifest_bytes = json.dumps([{"Config": "config.json", "Layers": ["layer/layer.tar"]}]).encode()

            with tarfile.open(image_tar, "w") as tf:
                ci = tarfile.TarInfo(name="config.json")
                ci.size = len(config_bytes)
                tf.addfile(ci, io.BytesIO(config_bytes))

                mi = tarfile.TarInfo(name="manifest.json")
                mi.size = len(manifest_bytes)
                tf.addfile(mi, io.BytesIO(manifest_bytes))

                tf.add(str(layer_tar), arcname="layer/layer.tar")

            dest = td_path / "extracted"
            dest.mkdir()
            _extract_from_tarball(image_tar, ["/app"], dest)

            extracted_file = dest / "app" / "model.py"
            assert extracted_file.exists()
            assert "import torch" in extracted_file.read_text()

            assert not (dest / "usr" / "bin" / "script.sh").exists()


class TestExtractSourceFromImage:
    @patch("aibom.scanners.container_extractor._find_binary")
    @patch("aibom.scanners.container_extractor._find_runtime")
    def test_no_runtime_no_tarball(self, mock_runtime, mock_binary):
        mock_runtime.return_value = None
        mock_binary.return_value = None
        result = extract_source_from_image("nonexistent:latest")
        assert result.error is not None
        assert "No container runtime" in result.error

    @patch("aibom.scanners.container_extractor._find_binary")
    @patch("aibom.scanners.container_extractor._find_runtime")
    def test_local_tarball_path(self, mock_runtime, mock_binary):
        mock_runtime.return_value = None
        mock_binary.return_value = None

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            layer_tar = td_path / "layer.tar"

            import io
            with tarfile.open(layer_tar, "w") as lt:
                content = b"from openai import OpenAI\n"
                info = tarfile.TarInfo(name="app/main.py")
                info.size = len(content)
                lt.addfile(info, io.BytesIO(content))

            image_tar = td_path / "image.tar"
            config_bytes = json.dumps({"config": {"WorkingDir": "/app"}}).encode()
            manifest_bytes = json.dumps([{"Config": "config.json", "Layers": ["layer/layer.tar"]}]).encode()

            with tarfile.open(image_tar, "w") as tf:
                ci = tarfile.TarInfo(name="config.json")
                ci.size = len(config_bytes)
                tf.addfile(ci, io.BytesIO(config_bytes))

                mi = tarfile.TarInfo(name="manifest.json")
                mi.size = len(manifest_bytes)
                tf.addfile(mi, io.BytesIO(manifest_bytes))

                tf.add(str(layer_tar), arcname="layer/layer.tar")

            out_dir = td_path / "out"
            result = extract_source_from_image(str(image_tar), output_dir=out_dir)
            assert result.error is None
            assert result.extracted_dir == out_dir
            assert result.config is not None
            assert result.config.workdir == "/app"
            assert "/app" in result.app_dirs
            assert result.tier_used == "python_tarball"
            assert (out_dir / "app" / "main.py").exists()

    @patch("aibom.scanners.container_extractor._find_binary")
    @patch("aibom.scanners.container_extractor._find_runtime")
    @patch("aibom.scanners.container_extractor._run_discovery")
    @patch("aibom.scanners.container_extractor._identify_app_dirs")
    @patch("aibom.scanners.container_extractor._run_extraction")
    def test_directory_only_extraction_reports_no_files(
        self,
        mock_extract,
        mock_identify,
        mock_discovery,
        mock_runtime,
        mock_binary,
    ):
        mock_runtime.return_value = None
        mock_binary.return_value = None
        mock_discovery.return_value = (ImageConfig(workdir="/app"), ["/app/main.py"])
        mock_identify.return_value = (["/app"], False, None)

        def _extract_only_dirs(*args, **kwargs):
            output_dir = args[3]
            (output_dir / "app").mkdir(parents=True, exist_ok=True)
            return True

        mock_extract.side_effect = _extract_only_dirs

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "out"
            result = extract_source_from_image("image.tar", output_dir=out_dir, tier="tarball")

        assert result.error == "Extraction succeeded but no files found"
        assert result.extracted_dir is None


class TestValidateTier:
    def test_all_valid_tiers_accepted(self):
        for t in VALID_TIERS:
            assert validate_tier(t) == t

    def test_case_insensitive(self):
        assert validate_tier("Docker") == "docker"
        assert validate_tier("TARBALL") == "tarball"

    def test_invalid_tier_raises(self):
        import typer
        with pytest.raises(typer.BadParameter, match="Unknown tier"):
            validate_tier("bogus")

    def test_valid_tiers_list(self):
        expected = {"auto", "syft", "docker", "podman", "nerdctl", "buildah", "crane", "skopeo", "tarball"}
        assert set(VALID_TIERS) == expected


class TestForcedTier:
    @patch("aibom.scanners.container_extractor._find_binary")
    @patch("aibom.scanners.container_extractor._find_runtime")
    def test_forced_tarball_with_file(self, mock_runtime, mock_binary):
        mock_runtime.return_value = None
        mock_binary.return_value = None

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            import io
            layer_tar = td_path / "layer.tar"
            with tarfile.open(layer_tar, "w") as lt:
                content = b"import openai\n"
                info = tarfile.TarInfo(name="app/main.py")
                info.size = len(content)
                lt.addfile(info, io.BytesIO(content))

            image_tar = td_path / "image.tar"
            config_bytes = json.dumps({"config": {"WorkingDir": "/app"}}).encode()
            manifest_bytes = json.dumps([{"Config": "config.json", "Layers": ["layer/layer.tar"]}]).encode()

            with tarfile.open(image_tar, "w") as tf:
                ci = tarfile.TarInfo(name="config.json")
                ci.size = len(config_bytes)
                tf.addfile(ci, io.BytesIO(config_bytes))
                mi = tarfile.TarInfo(name="manifest.json")
                mi.size = len(manifest_bytes)
                tf.addfile(mi, io.BytesIO(manifest_bytes))
                tf.add(str(layer_tar), arcname="layer/layer.tar")

            out_dir = td_path / "out"
            result = extract_source_from_image(str(image_tar), output_dir=out_dir, tier="tarball")
            assert result.error is None
            assert result.tier_used == "python_tarball"
            assert (out_dir / "app" / "main.py").exists()

    @patch("aibom.scanners.container_extractor._find_binary")
    @patch("aibom.scanners.container_extractor._find_runtime")
    def test_forced_tarball_without_file_errors(self, mock_runtime, mock_binary):
        mock_runtime.return_value = None
        mock_binary.return_value = None
        result = extract_source_from_image("nonexistent:latest", tier="tarball")
        assert result.error is not None
        assert "tarball" in result.error.lower()

    @patch("aibom.scanners.container_extractor._find_binary")
    @patch("aibom.scanners.container_extractor._find_runtime")
    def test_forced_docker_without_docker_errors(self, mock_runtime, mock_binary):
        mock_runtime.return_value = None
        mock_binary.return_value = None
        result = extract_source_from_image("some-image:latest", tier="docker")
        assert result.error is not None
        assert "docker" in result.error.lower()

    @patch("aibom.scanners.container_extractor._find_binary")
    @patch("aibom.scanners.container_extractor._find_runtime")
    def test_forced_syft_without_syft_errors(self, mock_runtime, mock_binary):
        mock_runtime.return_value = None
        mock_binary.return_value = None
        result = extract_source_from_image("some-image:latest", tier="syft")
        assert result.error is not None
        assert "syft" in result.error.lower()

    @patch("aibom.scanners.container_extractor._find_binary")
    @patch("aibom.scanners.container_extractor._find_runtime")
    def test_forced_skopeo_without_skopeo_errors(self, mock_runtime, mock_binary):
        mock_runtime.return_value = None
        mock_binary.return_value = None
        result = extract_source_from_image("some-image:latest", tier="skopeo")
        assert result.error is not None
        assert "skopeo" in result.error.lower()

    @patch("aibom.scanners.container_extractor._find_binary")
    @patch("aibom.scanners.container_extractor._find_runtime")
    def test_forced_buildah_on_macos_errors(self, mock_runtime, mock_binary):
        import platform
        if platform.system() != "Darwin":
            pytest.skip("buildah macOS guard only testable on macOS")

        mock_runtime.return_value = None
        mock_binary.return_value = "/usr/bin/buildah"
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            import io
            layer_tar = td_path / "layer.tar"
            with tarfile.open(layer_tar, "w") as lt:
                content = b"import torch\n"
                info = tarfile.TarInfo(name="app/train.py")
                info.size = len(content)
                lt.addfile(info, io.BytesIO(content))

            image_tar = td_path / "image.tar"
            config_bytes = json.dumps({"config": {"WorkingDir": "/app"}}).encode()
            manifest_bytes = json.dumps([{"Config": "config.json", "Layers": ["layer/layer.tar"]}]).encode()
            with tarfile.open(image_tar, "w") as tf:
                ci = tarfile.TarInfo(name="config.json")
                ci.size = len(config_bytes)
                tf.addfile(ci, io.BytesIO(config_bytes))
                mi = tarfile.TarInfo(name="manifest.json")
                mi.size = len(manifest_bytes)
                tf.addfile(mi, io.BytesIO(manifest_bytes))
                tf.add(str(layer_tar), arcname="layer/layer.tar")

            result = extract_source_from_image(str(image_tar), tier="buildah")
            assert result.error is not None
            assert "buildah" in result.error.lower() or result.tier_used != "buildah_mount"


def _build_ambiguous_image_tar(td_path: Path) -> Path:
    """Helper: build a tarball with 4+ app dirs and WORKDIR=/ to trigger needs_agentic."""
    layer_tar = td_path / "layer.tar"

    import io
    with tarfile.open(layer_tar, "w") as lt:
        for d in ("svcA", "svcB", "svcC", "svcD"):
            content = b"import openai\n"
            info = tarfile.TarInfo(name=f"{d}/main.py")
            info.size = len(content)
            lt.addfile(info, io.BytesIO(content))

    image_tar = td_path / "image.tar"
    config_bytes = json.dumps({"config": {"WorkingDir": "/"}}).encode()
    manifest_bytes = json.dumps(
        [{"Config": "config.json", "Layers": ["layer/layer.tar"]}]
    ).encode()

    import io as _io
    with tarfile.open(image_tar, "w") as tf:
        ci = tarfile.TarInfo(name="config.json")
        ci.size = len(config_bytes)
        tf.addfile(ci, _io.BytesIO(config_bytes))

        mi = tarfile.TarInfo(name="manifest.json")
        mi.size = len(manifest_bytes)
        tf.addfile(mi, _io.BytesIO(manifest_bytes))

        tf.add(str(layer_tar), arcname="layer/layer.tar")

    return image_tar


class TestAgenticLayoutResolution:
    """Verify that agentic layout resolution is invoked for ambiguous layouts."""

    @patch("aibom.scanners.container_extractor._find_binary")
    @patch("aibom.scanners.container_extractor._find_runtime")
    def test_agentic_narrows_dirs(self, mock_runtime, mock_binary):
        """When needs_agentic=True and llm_config present, agent is called."""
        mock_runtime.return_value = None
        mock_binary.return_value = None

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            image_tar = _build_ambiguous_image_tar(td_path)

            with patch(
                "aibom.agentic.agent.resolve_container_layout",
                return_value=["/svcA", "/svcB"],
            ) as mock_agent:
                out_dir = td_path / "out"
                result = extract_source_from_image(
                    str(image_tar),
                    output_dir=out_dir,
                    llm_config={"model": "test-model"},
                )
                mock_agent.assert_called_once()
                assert result.app_dirs == ["/svcA", "/svcB"]
                assert result.needs_agentic is False

    @patch("aibom.scanners.container_extractor._find_binary")
    @patch("aibom.scanners.container_extractor._find_runtime")
    def test_no_llm_config_skips_agentic(self, mock_runtime, mock_binary):
        """When no llm_config, ambiguous layout extracts all candidates."""
        mock_runtime.return_value = None
        mock_binary.return_value = None

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            image_tar = _build_ambiguous_image_tar(td_path)

            out_dir = td_path / "out"
            result = extract_source_from_image(str(image_tar), output_dir=out_dir)
            assert result.needs_agentic is True
            assert len(result.app_dirs) >= 4

    @patch("aibom.scanners.container_extractor._find_binary")
    @patch("aibom.scanners.container_extractor._find_runtime")
    def test_agentic_failure_falls_back(self, mock_runtime, mock_binary):
        """When agentic resolution fails, all candidates are used."""
        mock_runtime.return_value = None
        mock_binary.return_value = None

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            image_tar = _build_ambiguous_image_tar(td_path)

            with patch(
                "aibom.agentic.agent.resolve_container_layout",
                side_effect=RuntimeError("LLM unavailable"),
            ):
                out_dir = td_path / "out"
                result = extract_source_from_image(
                    str(image_tar),
                    output_dir=out_dir,
                    llm_config={"model": "test-model"},
                )
                assert len(result.app_dirs) >= 4
