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

"""Container source code extraction -- tiered Discovery ➜ Extraction pipeline.

Replaces the legacy ``container_utils.extract_app_from_docker`` which ran
``docker run -d --entrypoint sleep``.  The new approach never starts a container
process; it uses ``create``+``cp`` or direct tarball/mount access instead.

Supports multiple container runtimes (Docker, Podman, nerdctl, Buildah,
Skopeo, Crane) plus Anchore Syft for SBOM metadata, with a pure-Python
tarball fallback requiring zero external binaries.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image detection (moved from container_utils.py)
# ---------------------------------------------------------------------------

def is_container_image(source: str) -> bool:
    """Auto-detect whether *source* is a container image reference.

    Returns ``False`` for existing local files/directories.  Otherwise
    probes for a Docker-compatible runtime and attempts ``inspect`` (and
    ``pull`` if not found locally).
    """
    if os.path.isdir(source) or os.path.isfile(source):
        return False

    runtime = _find_runtime()
    if runtime is None:
        return False

    try:
        subprocess.run(
            [runtime.path, "image", "inspect", source],
            capture_output=True, check=True, timeout=10,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass

    _LOGGER.info("Image '%s' not found locally, attempting pull", source)
    try:
        subprocess.run(
            [runtime.path, "pull", source],
            capture_output=True, check=True, timeout=120,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


_SOURCE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".ipynb", ".js", ".ts", ".jsx", ".tsx", ".mjs",
    ".java", ".go", ".rs", ".rb", ".cs", ".php",
    ".yaml", ".yml", ".json", ".toml", ".env", ".tf", ".hcl",
})

_SYSTEM_PATHS: tuple[str, ...] = (
    "/usr/share", "/usr/bin", "/usr/sbin", "/usr/lib",
    "/var", "/etc", "/bin", "/sbin", "/lib", "/lib64",
    "/proc", "/sys", "/dev", "/root/.cache", "/tmp", "/run",
)

_DEPENDENCY_PATHS: tuple[str, ...] = (
    "site-packages", "dist-packages", "node_modules", "vendor/bundle",
)

_FALLBACK_APP_DIRS: tuple[str, ...] = (
    "/app", "/opt", "/srv", "/workspace", "/code",
)

_MANIFEST_PATTERNS: tuple[str, ...] = (
    "requirements.txt", "pyproject.toml", "package.json", "go.mod",
    "pom.xml", "Cargo.toml", "Gemfile", "build.gradle",
)


@dataclass
class _RuntimeInfo:
    name: str
    path: str


@dataclass
class ImageConfig:
    workdir: str = "/"
    entrypoint: list[str] = field(default_factory=list)
    cmd: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    image_ref: str
    extracted_dir: Optional[Path] = None
    config: Optional[ImageConfig] = None
    app_dirs: list[str] = field(default_factory=list)
    error: Optional[str] = None
    tier_used: str = "none"
    needs_agentic: bool = False
    agentic_hint: str = ""
    # Multi-source correlation metadata. ``sbom_packages`` holds
    # package URLs (purls) discovered by Syft during extraction, used for
    # SBOM-backed shared-dependency correlation. ``base_image`` and
    # ``source_repo_url`` are surfaced from OCI image labels
    # (``org.opencontainers.image.base.name`` /
    # ``org.opencontainers.image.source``) for base-image lineage and
    # derived-from-repo advisories. Populated best-effort; empty when the
    # discovery tier does not provide them (e.g. Syft absent → no SBOM).
    sbom_packages: list[str] = field(default_factory=list)
    base_image: str = ""
    source_repo_url: str = ""


def _find_runtime() -> Optional[_RuntimeInfo]:
    """Probe for a Docker-compatible CLI in preference order."""
    for name in ("docker", "podman", "nerdctl"):
        path = shutil.which(name)
        if path:
            if name == "docker":
                try:
                    subprocess.run(
                        [path, "info"],
                        capture_output=True, timeout=5, check=True,
                    )
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                    continue
            return _RuntimeInfo(name, path)
    return None


def _find_binary(name: str) -> Optional[str]:
    return shutil.which(name)


def _parse_image_config(config_json: dict | str) -> ImageConfig:
    if isinstance(config_json, str):
        import base64
        try:
            config_json = json.loads(base64.b64decode(config_json))
        except Exception:
            try:
                config_json = json.loads(config_json)
            except Exception:
                return ImageConfig()

    cfg = config_json.get("config") or config_json.get("Config") or {}
    workdir = cfg.get("WorkingDir") or "/"
    entrypoint = cfg.get("Entrypoint") or []
    cmd = cfg.get("Cmd") or []
    env_list = cfg.get("Env") or []
    env_dict: dict[str, str] = {}
    for entry in env_list:
        if isinstance(entry, str) and "=" in entry:
            k, _, v = entry.partition("=")
            env_dict[k] = v
    labels_raw = cfg.get("Labels") or {}
    labels: dict[str, str] = {}
    if isinstance(labels_raw, dict):
        for k, v in labels_raw.items():
            if isinstance(k, str):
                labels[k] = "" if v is None else str(v)
    return ImageConfig(
        workdir=workdir,
        entrypoint=entrypoint,
        cmd=cmd,
        env=env_dict,
        labels=labels,
    )


def _entrypoint_target_dir(config: ImageConfig) -> Optional[str]:
    """Parse ENTRYPOINT/CMD to find the directory of the main script."""
    combined = config.entrypoint + config.cmd
    for token in combined:
        if token.startswith("-"):
            continue
        if "/" in token and "." in token.split("/")[-1]:
            parent = "/".join(token.split("/")[:-1])
            if parent and not any(parent.startswith(sp) for sp in _SYSTEM_PATHS):
                return parent
    return None


def _is_system_path(path: str) -> bool:
    return any(path.startswith(sp) for sp in _SYSTEM_PATHS)


def _is_dependency_path(path: str) -> bool:
    return any(dp in path for dp in _DEPENDENCY_PATHS)


def _identify_app_dirs(
    config: ImageConfig,
    file_listing: Sequence[str],
) -> tuple[list[str], bool, str]:
    """Deterministic identification of app code directories.

    Returns (app_dirs, needs_agentic, hint).
    """
    candidates: set[str] = set()

    if config.workdir and config.workdir != "/":
        candidates.add(config.workdir.rstrip("/"))

    ep_dir = _entrypoint_target_dir(config)
    if ep_dir:
        candidates.add(ep_dir)

    source_dirs: set[str] = set()
    for fpath in file_listing:
        if _is_system_path(fpath) or _is_dependency_path(fpath):
            continue
        ext = os.path.splitext(fpath)[1].lower()
        if ext in _SOURCE_EXTENSIONS:
            parts = fpath.strip("/").split("/")
            if len(parts) > 1:
                source_dirs.add("/" + parts[0])

    candidates.update(source_dirs)
    if not candidates:
        for fb in _FALLBACK_APP_DIRS:
            if any(f.startswith(fb) for f in file_listing):
                candidates.add(fb)

    app_dirs = sorted(candidates)
    needs_agentic = False
    hint = ""
    if len(app_dirs) > 3 or (config.workdir == "/" and not ep_dir):
        needs_agentic = True
        hint = (
            f"Ambiguous app layout: WORKDIR={config.workdir}, "
            f"{len(app_dirs)} candidate dirs. "
            "Agent should determine which contain app code."
        )

    return app_dirs, needs_agentic, hint


# ---------------------------------------------------------------------------
# Tier 1: Syft discovery
# ---------------------------------------------------------------------------

def _sbom_packages_from_syft(data: dict) -> list[str]:
    """Extract package URLs (purls) from a Syft JSON catalog.

    Falls back to a ``type/name@version`` synthetic identifier when an
    artifact has no purl, so SBOM-backed correlation still has a stable
    key. Duplicates are removed while preserving first-seen order.
    """
    seen: set[str] = set()
    out: list[str] = []
    for art in data.get("artifacts", []):
        if not isinstance(art, dict):
            continue
        purl = art.get("purl")
        if not purl:
            name = art.get("name") or ""
            if not name:
                continue
            version = art.get("version") or ""
            atype = art.get("type") or "pkg"
            purl = f"{atype}/{name}@{version}" if version else f"{atype}/{name}"
        if purl not in seen:
            seen.add(purl)
            out.append(purl)
    return out


def _discover_with_syft(
    image_ref: str, syft_path: str
) -> tuple[ImageConfig, list[str], list[str]]:
    result = subprocess.run(
        [syft_path, image_ref, "-o", "syft-json", "-q"],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Syft failed: {result.stderr[:500]}")
    data = json.loads(result.stdout)

    source_meta = data.get("source", {}).get("metadata", {})
    config_raw = source_meta.get("config", {})
    config = _parse_image_config(config_raw)

    file_listing: list[str] = []
    for f in data.get("files", []):
        loc = f.get("location", {})
        path = loc.get("path") or loc.get("realPath", "")
        if path:
            file_listing.append(path)

    sbom_packages = _sbom_packages_from_syft(data)

    return config, file_listing, sbom_packages


# ---------------------------------------------------------------------------
# Tier 2: Docker/Podman/nerdctl discovery via save + tarball parse
# ---------------------------------------------------------------------------

def _discover_with_runtime_save(image_ref: str, runtime: _RuntimeInfo) -> tuple[ImageConfig, list[str]]:
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        subprocess.run(
            [runtime.path, "save", "-o", tmp_path, image_ref],
            capture_output=True, timeout=300, check=True,
        )
        return _discover_from_tarball(Path(tmp_path))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _discover_from_tarball(tar_path: Path) -> tuple[ImageConfig, list[str]]:
    """Parse a docker-save tarball for config + file listing.

    Handles both legacy Docker format (``<hash>/layer.tar``) and OCI
    format (``blobs/sha256/<hash>``).
    """
    config = ImageConfig()
    file_listing: list[str] = []

    with tarfile.open(tar_path, "r") as tf:
        manifest_f = tf.extractfile("manifest.json")
        if not manifest_f:
            raise RuntimeError("No manifest.json in tarball")
        manifest = json.loads(manifest_f.read())
        if not isinstance(manifest, list) or not manifest:
            raise RuntimeError("Empty manifest")

        config_name = manifest[0].get("Config", "")
        if config_name:
            cf = tf.extractfile(config_name)
            if cf:
                config = _parse_image_config(json.loads(cf.read()))

        layer_names = manifest[0].get("Layers", [])

        for layer_name in layer_names:
            try:
                layer_f = tf.extractfile(layer_name)
            except KeyError:
                continue
            if not layer_f:
                continue
            try:
                with tarfile.open(fileobj=layer_f) as lf:
                    for lm in lf.getmembers():
                        if lm.isfile():
                            file_listing.append("/" + lm.name.lstrip("./"))
            except (tarfile.TarError, EOFError):
                _LOGGER.debug("Corrupt layer: %s", layer_name)

    return config, file_listing


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _extract_with_runtime_create(
    image_ref: str,
    runtime: _RuntimeInfo,
    app_dirs: list[str],
    dest: Path,
) -> None:
    """Use ``create`` + ``cp`` + ``rm`` — no process is started."""
    result = subprocess.run(
        [runtime.path, "create", image_ref, "/bin/true"],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{runtime.name} create failed: {result.stderr[:500]}")
    container_id = result.stdout.strip()

    try:
        for app_dir in app_dirs:
            target = dest / app_dir.strip("/")
            target.mkdir(parents=True, exist_ok=True)
            src_path = f"{container_id}:{app_dir}/."
            cp_result = subprocess.run(
                [runtime.path, "cp", src_path, str(target)],
                capture_output=True, timeout=120,
            )
            if cp_result.returncode != 0:
                _LOGGER.debug(
                    "%s cp failed for %s: %s",
                    runtime.name, app_dir,
                    cp_result.stderr.decode("utf-8", errors="replace")[:300],
                )
    finally:
        subprocess.run(
            [runtime.path, "rm", container_id],
            capture_output=True, timeout=30,
        )


def _extract_with_buildah(
    image_ref: str,
    buildah_path: str,
    app_dirs: list[str],
    dest: Path,
) -> None:
    """Use ``buildah from`` + ``buildah mount`` for direct filesystem access.

    On macOS buildah only runs inside the Podman Machine VM, so we
    detect that situation and fall through to let other tiers handle it.
    On Linux, buildah mount requires either root or ``buildah unshare``.
    """
    import platform

    if platform.system() == "Darwin":
        raise RuntimeError(
            "buildah mount requires Linux; on macOS it runs inside the "
            "Podman VM and cannot mount to the host filesystem"
        )

    result = subprocess.run(
        [buildah_path, "from", image_ref],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"buildah from failed: {result.stderr[:500]}")
    container = result.stdout.strip()

    try:
        mount_result = subprocess.run(
            [buildah_path, "unshare", "--", "sh", "-c",
             f"{shlex.quote(buildah_path)} mount {shlex.quote(container)}"],
            capture_output=True, text=True, timeout=30,
        )
        if mount_result.returncode != 0:
            mount_result = subprocess.run(
                [buildah_path, "mount", container],
                capture_output=True, text=True, timeout=30,
            )
        if mount_result.returncode != 0:
            raise RuntimeError(f"buildah mount failed: {mount_result.stderr[:500]}")
        mount_point = Path(mount_result.stdout.strip())

        for app_dir in app_dirs:
            src = mount_point / app_dir.strip("/")
            if src.exists():
                target = dest / app_dir.strip("/")
                shutil.copytree(str(src), str(target), dirs_exist_ok=True)

        subprocess.run(
            [buildah_path, "unmount", container],
            capture_output=True, timeout=30,
        )
    finally:
        subprocess.run(
            [buildah_path, "rm", container],
            capture_output=True, timeout=30,
        )


def _extract_with_crane(
    image_ref: str,
    crane_path: str,
    app_dirs: list[str],
    dest: Path,
    *,
    tarball_path: Optional[Path] = None,
) -> None:
    """Use ``crane export`` piped to tar extraction.

    Crane only talks to registries, not to the local Docker daemon.
    When the image is local-only we pipe a saved tarball via stdin::

        crane export - - < saved.tar | tar -xf - app/

    If *tarball_path* is ``None`` and a container runtime is available
    we create a temporary tarball automatically.
    """
    need_stdin = False
    if tarball_path is None:
        probe = subprocess.run(
            [crane_path, "config", image_ref],
            capture_output=True, timeout=30,
        )
        if probe.returncode != 0:
            runtime = _find_runtime()
            if runtime:
                tmp_fd = tempfile.NamedTemporaryFile(suffix=".tar", delete=False)
                tmp_fd.close()
                tmp_tar = Path(tmp_fd.name)
                subprocess.run(
                    [runtime.path, "save", "-o", str(tmp_tar), image_ref],
                    capture_output=True, timeout=300, check=True,
                )
                tarball_path = tmp_tar
                need_stdin = True
            else:
                raise RuntimeError(
                    f"crane cannot reach {image_ref} in any registry "
                    "and no runtime is available to save a tarball"
                )
    else:
        need_stdin = True

    try:
        for app_dir in app_dirs:
            target = dest / app_dir.strip("/")
            target.mkdir(parents=True, exist_ok=True)
            stripped = app_dir.strip("/")

            if need_stdin and tarball_path:
                cmd = (
                    f"{shlex.quote(crane_path)} export - - "
                    f"< {shlex.quote(str(tarball_path))} | "
                    f"tar -xf - -C {shlex.quote(str(target))} "
                    f"--strip-components=1 {shlex.quote(stripped + '/')}"
                )
            else:
                cmd = (
                    f"{shlex.quote(crane_path)} export "
                    f"{shlex.quote(image_ref)} - | "
                    f"tar -xf - -C {shlex.quote(str(target))} "
                    f"--strip-components=1 {shlex.quote(stripped + '/')}"
                )

            result = subprocess.run(
                cmd, shell=True, capture_output=True, timeout=300,
            )
            if result.returncode != 0:
                _LOGGER.debug(
                    "crane export failed for %s: %s",
                    app_dir, result.stderr[:300],
                )
    finally:
        if need_stdin and tarball_path and str(tarball_path) != image_ref:
            try:
                tarball_path.unlink()
            except OSError:
                pass


def _extract_from_tarball(
    tar_path: Path,
    app_dirs: list[str],
    dest: Path,
) -> None:
    """Pure-Python extraction from a docker-save tarball.

    Handles both legacy Docker format (``<hash>/layer.tar``) and OCI
    format (``blobs/sha256/<hash>``).
    """
    prefix_set = tuple(d.strip("/") + "/" for d in app_dirs)

    with tarfile.open(tar_path, "r") as tf:
        manifest_f = tf.extractfile("manifest.json")
        if not manifest_f:
            return
        manifest = json.loads(manifest_f.read())
        if not isinstance(manifest, list) or not manifest:
            return
        layer_names = manifest[0].get("Layers", [])

        for layer_name in layer_names:
            try:
                layer_f = tf.extractfile(layer_name)
            except KeyError:
                continue
            if not layer_f:
                continue
            try:
                with tarfile.open(fileobj=layer_f) as lf:
                    for lm in lf.getmembers():
                        clean = lm.name.lstrip("./")
                        if not any(clean.startswith(p) for p in prefix_set):
                            continue
                        if lm.isfile():
                            out_path = dest / clean
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            ef = lf.extractfile(lm)
                            if ef:
                                out_path.write_bytes(ef.read())
            except (tarfile.TarError, EOFError):
                continue


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class _TierMissing(Exception):
    """Raised when a forced tier's required tool is not available."""


def _run_discovery(
    tier: str,
    image_ref: str,
    syft_path: Optional[str],
    runtime: Optional[_RuntimeInfo],
    result: "ExtractionResult",
    _require,
    _require_runtime,
) -> tuple[Optional["ImageConfig"], list[str]]:
    """Run the discovery phase respecting *tier*."""
    config: Optional[ImageConfig] = None
    file_listing: list[str] = []

    if tier in ("auto", "syft"):
        if tier == "syft":
            _require("syft", syft_path)
        if syft_path:
            try:
                _LOGGER.info("Container discovery: using Syft")
                config, file_listing, sbom_packages = _discover_with_syft(
                    image_ref, syft_path
                )
                result.sbom_packages = sbom_packages
                result.tier_used = "syft"
            except Exception:
                if tier == "syft":
                    raise
                _LOGGER.warning("Syft discovery failed, falling back", exc_info=True)

    if tier in ("auto",) and config is None and runtime:
        try:
            _LOGGER.info("Container discovery: using %s save", runtime.name)
            config, file_listing = _discover_with_runtime_save(image_ref, runtime)
            result.tier_used = f"{runtime.name}_save"
        except Exception:
            _LOGGER.warning("%s save discovery failed", runtime.name, exc_info=True)

    if tier in ("docker", "podman", "nerdctl") and config is None:
        forced_rt = _require_runtime(tier)
        try:
            _LOGGER.info("Container discovery: using %s save", forced_rt.name)
            config, file_listing = _discover_with_runtime_save(image_ref, forced_rt)
            result.tier_used = f"{forced_rt.name}_save"
        except Exception:
            raise

    if tier in ("auto", "skopeo") and config is None:
        skopeo = _find_binary("skopeo")
        if tier == "skopeo":
            _require("skopeo", skopeo)
        if skopeo:
            tmp_fd = tempfile.NamedTemporaryFile(suffix=".tar", delete=False)
            tmp_fd.close()
            tmp_tar = Path(tmp_fd.name)
            try:
                _LOGGER.info("Container discovery: using skopeo copy → tarball")
                subprocess.run(
                    [skopeo, "copy", f"docker-daemon:{image_ref}",
                     f"docker-archive:{tmp_tar}"],
                    capture_output=True, timeout=300, check=True,
                )
                config, file_listing = _discover_from_tarball(tmp_tar)
                result.tier_used = "skopeo"
            except Exception:
                if tier == "skopeo":
                    raise
                _LOGGER.warning("skopeo discovery failed", exc_info=True)
            finally:
                tmp_tar.unlink(missing_ok=True)

    if tier in ("auto", "tarball", "crane", "buildah") and config is None:
        if Path(image_ref).exists():
            try:
                _LOGGER.info("Container discovery: parsing local tarball")
                config, file_listing = _discover_from_tarball(Path(image_ref))
                result.tier_used = "tarball"
            except Exception:
                if tier == "tarball":
                    raise
                _LOGGER.warning("Tarball discovery failed", exc_info=True)
        elif tier in ("crane", "buildah"):
            if runtime:
                _LOGGER.info("Container discovery: using %s save (for %s tier)", runtime.name, tier)
                config, file_listing = _discover_with_runtime_save(image_ref, runtime)
                result.tier_used = f"{runtime.name}_save"
            else:
                result.error = f"--container-extraction-tier={tier} needs a tarball or a runtime for discovery"
                raise _TierMissing(result.error)
        elif tier == "tarball":
            result.error = f"--container-extraction-tier=tarball requires image_ref to be a .tar file path"
            raise _TierMissing(result.error)

    return config, file_listing


def _run_extraction(
    tier: str,
    image_ref: str,
    app_dirs: list[str],
    output_dir: Path,
    result: "ExtractionResult",
    runtime: Optional[_RuntimeInfo],
    buildah_path: Optional[str],
    crane_path: Optional[str],
    _require,
    _require_runtime,
) -> bool:
    """Run the extraction phase respecting *tier*."""
    extracted = False

    if tier in ("auto", "docker", "podman", "nerdctl", "syft"):
        rt = runtime
        if tier in ("docker", "podman", "nerdctl"):
            rt = _require_runtime(tier)
        elif tier == "syft" and runtime:
            rt = runtime
        if rt:
            try:
                _LOGGER.info("Container extraction: %s create+cp", rt.name)
                _extract_with_runtime_create(image_ref, rt, app_dirs, output_dir)
                result.tier_used = f"{rt.name}_create_cp"
                extracted = True
            except Exception as exc:
                if tier in ("docker", "podman", "nerdctl"):
                    result.error = f"{tier} extraction failed: {exc}"
                    return False
                _LOGGER.warning("%s create+cp failed", rt.name, exc_info=True)

    if tier in ("auto", "buildah") and not extracted:
        bp = buildah_path
        if tier == "buildah":
            bp = _require("buildah", buildah_path)
        if bp:
            try:
                _LOGGER.info("Container extraction: buildah mount")
                _extract_with_buildah(image_ref, bp, app_dirs, output_dir)
                result.tier_used = "buildah_mount"
                extracted = True
            except Exception as exc:
                if tier == "buildah":
                    result.error = f"buildah extraction failed: {exc}"
                    return False
                _LOGGER.warning("buildah mount failed", exc_info=True)

    if tier in ("auto", "crane") and not extracted:
        cp = crane_path
        if tier == "crane":
            cp = _require("crane", crane_path)
        if cp:
            try:
                _LOGGER.info("Container extraction: crane export")
                _extract_with_crane(image_ref, cp, app_dirs, output_dir)
                result.tier_used = "crane_export"
                extracted = True
            except Exception as exc:
                if tier == "crane":
                    result.error = f"crane extraction failed: {exc}"
                    return False
                _LOGGER.warning("crane export failed", exc_info=True)

    if tier in ("auto", "skopeo") and not extracted:
        skopeo = _find_binary("skopeo")
        if tier == "skopeo":
            skopeo = _require("skopeo", skopeo)
        if skopeo:
            tmp_fd = tempfile.NamedTemporaryFile(suffix=".tar", delete=False)
            tmp_fd.close()
            tmp_tar = Path(tmp_fd.name)
            try:
                _LOGGER.info("Container extraction: skopeo copy → tarball → Python extract")
                subprocess.run(
                    [skopeo, "copy", f"docker-daemon:{image_ref}",
                     f"docker-archive:{tmp_tar}"],
                    capture_output=True, timeout=300, check=True,
                )
                _extract_from_tarball(tmp_tar, app_dirs, output_dir)
                result.tier_used = "skopeo_tarball"
                extracted = True
            except Exception as exc:
                if tier == "skopeo":
                    result.error = f"skopeo extraction failed: {exc}"
                    return False
                _LOGGER.warning("skopeo extraction failed", exc_info=True)
            finally:
                tmp_tar.unlink(missing_ok=True)

    if tier in ("auto", "tarball") and not extracted:
        tar_path: Optional[Path] = None
        if Path(image_ref).exists():
            tar_path = Path(image_ref)
        elif runtime:
            _LOGGER.info("Saving image to tarball for extraction")
            tmp_fd = tempfile.NamedTemporaryFile(suffix=".tar", delete=False)
            tmp_fd.close()
            tmp_tar = Path(tmp_fd.name)
            try:
                subprocess.run(
                    [runtime.path, "save", "-o", str(tmp_tar), image_ref],
                    capture_output=True, timeout=300, check=True,
                )
                tar_path = tmp_tar
            except Exception:
                _LOGGER.warning("Save-for-extraction failed", exc_info=True)

        if tar_path:
            try:
                _LOGGER.info("Container extraction: pure Python tarball")
                _extract_from_tarball(tar_path, app_dirs, output_dir)
                result.tier_used = "python_tarball"
                extracted = True
            except Exception:
                _LOGGER.warning("Tarball extraction failed", exc_info=True)
            finally:
                if str(tar_path) != image_ref:
                    try:
                        tar_path.unlink()
                    except OSError:
                        pass

    return extracted


VALID_TIERS: tuple[str, ...] = (
    "auto",
    "syft",
    "docker",
    "podman",
    "nerdctl",
    "buildah",
    "crane",
    "skopeo",
    "tarball",
)


def validate_tier(tier: str) -> str:
    """Return *tier* if valid, otherwise raise ``typer.BadParameter``."""
    t = tier.lower().strip()
    if t not in VALID_TIERS:
        import typer
        raise typer.BadParameter(
            f"Unknown tier '{tier}'. Choose from: {', '.join(VALID_TIERS)}"
        )
    return t


def extract_source_from_image(
    image_ref: str,
    output_dir: Optional[Path] = None,
    *,
    llm_config: Optional[dict[str, str]] = None,
    tier: str = "auto",
) -> ExtractionResult:
    """Extract application source code from a container image.

    *tier* controls which tools are used.  ``"auto"`` (default) cascades
    through available tools.  Specific values force a single tool:

    ========  ===========================  ============================
    Tier      Customer prerequisite        What the code does
    ========  ===========================  ============================
    auto      (any available tool)         Best-effort cascade
    syft      ``syft`` on PATH             Syft JSON catalog → runtime cp
    docker    Docker daemon running        docker save → docker create+cp
    podman    Podman daemon running        podman save → podman create+cp
    nerdctl   nerdctl + containerd         nerdctl save → nerdctl create+cp
    buildah   buildah on Linux             buildah from → buildah mount+cp
    crane     ``crane`` on PATH + runtime  crane export (piped from tarball)
    skopeo    ``skopeo`` on PATH           skopeo copy → Python tarball parse
    tarball   image_ref is a ``.tar`` file Pure Python layer parsing
    ========  ===========================  ============================

    Returns an :class:`ExtractionResult` with the extracted directory path,
    image config, and metadata.
    """
    tier = tier.lower().strip()
    result = ExtractionResult(image_ref=image_ref)

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="aibom_container_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Resolve available tools ---
    syft_path = _find_binary("syft")
    if not syft_path:
        aibom_syft = Path.home() / ".aibom" / "bin" / "syft"
        if aibom_syft.is_file():
            syft_path = str(aibom_syft)
    runtime = _find_runtime()
    buildah_path = _find_binary("buildah")
    crane_path = _find_binary("crane")

    def _require(tool_name: str, path: Optional[str]) -> str:
        if not path:
            result.error = f"--container-extraction-tier={tier} requires '{tool_name}' but it was not found on PATH"
            raise _TierMissing(result.error)
        return path

    def _require_runtime(name: str) -> _RuntimeInfo:
        if runtime and runtime.name == name:
            return runtime
        forced_path = _find_binary(name)
        if forced_path:
            return _RuntimeInfo(name, forced_path)
        result.error = f"--container-extraction-tier={tier} requires '{name}' but it was not found or not running"
        raise _TierMissing(result.error)

    # --- Discovery ---
    config: Optional[ImageConfig] = None
    file_listing: list[str] = []

    try:
        config, file_listing = _run_discovery(
            tier, image_ref, syft_path, runtime, result,
            _require, _require_runtime,
        )
    except _TierMissing:
        return result
    except Exception as exc:
        if tier != "auto":
            result.error = f"{tier} discovery failed: {exc}"
            return result
        raise

    if config is None:
        if tier == "auto":
            result.error = "No container runtime available and image is not a local tarball"
        return result

    result.config = config
    # Surface OCI base-image lineage + upstream source repo from standard
    # image annotations (present regardless of discovery tier, since every
    # tier parses the image config). Used by cross-source correlation for
    # SHARED_BASE_IMAGE links and the unscanned-upstream-repo advisory.
    result.base_image = config.labels.get("org.opencontainers.image.base.name", "")
    result.source_repo_url = config.labels.get("org.opencontainers.image.source", "")
    app_dirs, needs_agentic, hint = _identify_app_dirs(config, file_listing)
    result.needs_agentic = needs_agentic
    result.agentic_hint = hint

    if not app_dirs:
        result.error = "Could not identify application directories in image"
        return result

    _LOGGER.info(
        "Identified %d candidate app directories in %s: %s",
        len(app_dirs), image_ref, app_dirs,
    )

    # --- Agentic layout resolution (when ambiguous) ---
    if needs_agentic and llm_config and llm_config.get("model"):
        _LOGGER.info(
            "Ambiguous layout detected (%s), invoking agentic resolution", hint,
        )
        try:
            from ..agentic.agent import resolve_container_layout

            refined = resolve_container_layout(
                model_string=llm_config["model"],
                image_config={
                    "workdir": config.workdir,
                    "entrypoint": config.entrypoint,
                    "cmd": config.cmd,
                    "env": dict(config.env),
                },
                candidate_dirs=app_dirs,
                file_listing=file_listing,
                llm_config=llm_config,
            )
            if refined:
                _LOGGER.info(
                    "Agentic resolution narrowed %d → %d dirs: %s",
                    len(app_dirs), len(refined), refined,
                )
                app_dirs = refined
                result.needs_agentic = False
        except ImportError:
            _LOGGER.info(
                "Agentic extras not installed, extracting all %d candidate dirs",
                len(app_dirs),
            )
        except Exception:
            _LOGGER.warning(
                "Agentic layout resolution failed, using all candidates",
                exc_info=True,
            )
    elif needs_agentic:
        _LOGGER.info(
            "Ambiguous layout but no LLM configured — extracting all %d candidate dirs. "
            "Re-run with --llm-model to enable agentic resolution.",
            len(app_dirs),
        )

    result.app_dirs = app_dirs

    # --- Extraction ---
    try:
        extracted = _run_extraction(
            tier, image_ref, app_dirs, output_dir, result,
            runtime, buildah_path, crane_path,
            _require, _require_runtime,
        )
    except _TierMissing:
        return result
    except Exception as exc:
        if tier != "auto":
            result.error = f"{tier} extraction failed: {exc}"
            return result
        raise

    if not extracted:
        result.error = (
            f"Extraction tier '{tier}' failed"
            if tier != "auto" else "All extraction tiers failed"
        )
        return result

    has_files = any(path.is_file() for path in output_dir.rglob("*"))
    if not has_files:
        result.error = "Extraction succeeded but no files found"
        return result

    result.extracted_dir = output_dir
    return result
