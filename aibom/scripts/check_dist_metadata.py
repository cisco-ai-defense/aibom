#!/usr/bin/env python3
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

"""Check that built distributions carry the Apache-2.0 license metadata and file.

Usage (from aibom/, after ``uv build``):
    python scripts/check_dist_metadata.py [dist]
"""

import sys
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

EXPECTED_LICENSE = "Apache-2.0"


def check_metadata(name: str, metadata: bytes, license_present: bool) -> list[str]:
    fields = BytesParser().parsebytes(metadata)
    errors = []
    if fields.get("License-Expression") != EXPECTED_LICENSE:
        errors.append(f"{name}: License-Expression is not {EXPECTED_LICENSE}")
    if "LICENSE" not in fields.get_all("License-File", []):
        errors.append(f"{name}: no License-File: LICENSE entry")
    if not license_present:
        errors.append(f"{name}: LICENSE file is not bundled")
    return errors


def check_wheel(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        metadata = [n for n in names if n.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            return [f"{path.name}: expected one .dist-info/METADATA"]
        dist_info = metadata[0].rsplit("/", 1)[0]
        license_present = f"{dist_info}/licenses/LICENSE" in names
        return check_metadata(path.name, wheel.read(metadata[0]), license_present)


def check_sdist(path: Path) -> list[str]:
    with tarfile.open(path) as sdist:
        names = sdist.getnames()
        root = names[0].split("/", 1)[0]
        if f"{root}/PKG-INFO" not in names:
            return [f"{path.name}: no top-level PKG-INFO"]
        metadata = sdist.extractfile(f"{root}/PKG-INFO").read()
        return check_metadata(path.name, metadata, f"{root}/LICENSE" in names)


def main(dist_dir: str = "dist") -> int:
    wheels = sorted(Path(dist_dir).glob("*.whl"))
    sdists = sorted(Path(dist_dir).glob("*.tar.gz"))
    if not wheels or not sdists:
        print(f"expected a wheel and an sdist in {dist_dir}/", file=sys.stderr)
        return 1
    errors = [e for p in wheels for e in check_wheel(p)]
    errors += [e for p in sdists for e in check_sdist(p)]
    for error in errors:
        print(error, file=sys.stderr)
    if not errors:
        print(f"license metadata OK: {len(wheels) + len(sdists)} distribution(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
