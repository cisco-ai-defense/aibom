# Container Scanning Guide

Cisco AI BOM can analyze container images by extracting application source code and running the full scan pipeline against it. The CLI auto-detects container image references (anything that isn't an existing local file or directory) and chooses the best available extraction method.

## How It Works

Container scanning follows a Discovery-then-Extraction pipeline:

1. **Detection** — The CLI determines whether a source argument is a container image reference by checking if it exists as a local path. If not, it probes for a container runtime and attempts to inspect/pull the image.
2. **Discovery** — The available extraction tools are probed in priority order (or a specific tier is forced via `--container-extraction-tier`).
3. **Extraction** — Application source code is extracted from the image to a temporary directory. No container process is started — the extractor uses `create`+`cp`, direct tarball access, or runtime-specific export commands.
4. **SBOM enrichment** — If Anchore Syft is available, it generates an SBOM for the image, providing package metadata that enriches the scan results.
5. **Scanning** — The extracted source code is scanned using the same pipeline as local directories.
6. **Agentic layout resolution** — If the extracted directory structure is ambiguous (e.g., multiple application roots), agentic reasoning can help identify the correct application code paths.

## Supported Extraction Tiers

| Tier | Tool | How It Extracts |
|------|------|-----------------|
| `docker` | Docker CLI | `docker create` + `docker cp` (no process started). |
| `podman` | Podman | `podman create` + `podman cp`. |
| `nerdctl` | nerdctl | `nerdctl create` + `nerdctl cp`. |
| `buildah` | Buildah | `buildah from` (creates a working container) + `buildah mount` to access the filesystem. |
| `skopeo` | Skopeo | `skopeo copy` to an OCI layout directory, then extract layers from the tarball. |
| `crane` | Crane | `crane export` to a tar stream, then extract from the tarball. |
| `syft` | Anchore Syft | SBOM-only metadata extraction (no source code extraction). Useful when only package-level inventory is needed. |
| `tarball` | Python stdlib | Pure-Python fallback: `docker save` / `podman save` piped to `tarfile` extraction. Requires a Docker-compatible runtime for the save step. |

### Auto-detection priority

When `--container-extraction-tier auto` (the default), the CLI probes tools in this order:

1. Docker
2. Podman
3. nerdctl
4. Buildah
5. Skopeo
6. Crane

The first available tool is used. If none are found, the pure-Python tarball fallback is attempted.

## Usage

### Basic container scan

```bash
cisco-aibom analyze my-app:latest -o json -O report.json
```

The CLI auto-detects the image reference and extracts source code using the best available tool.

### Force a specific tier

```bash
cisco-aibom analyze my-app:latest -o json -O report.json --container-extraction-tier podman
```

Valid values: `auto`, `syft`, `docker`, `podman`, `nerdctl`, `buildah`, `crane`, `skopeo`, `tarball`.

### Scan multiple images from a file

```bash
# images.json: ["app1:latest", "app2:v1.2", "app3:prod"]
cisco-aibom analyze --images-file images.json -o json -O report.json
```

### Container scan with agentic enrichment

```bash
cisco-aibom analyze my-app:latest -o json -O report.json \
  --llm-model gpt-4o --llm-provider openai --llm-api-key $OPENAI_API_KEY
```

## Syft Integration

When [Anchore Syft](https://github.com/anchore/syft) is installed, the CLI automatically runs it against the container image to collect SBOM metadata (packages, versions, licenses). This metadata enriches the scan results regardless of which extraction tier is used for source code.

Install Syft:

```bash
curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
```

Syft is not required — if it's not available, the CLI proceeds with source-code-only analysis.

## What Gets Extracted

The extractor looks for application source code in these directories (in priority order):

1. `/app` — Common convention for containerized applications.
2. `/workspace`, `/src`, `/code` — Alternative app directory conventions.
3. Python `site-packages` — Installed Python packages.
4. The image's `WORKDIR` — As declared in the Dockerfile.

If the layout is ambiguous (multiple candidate directories), and agentic mode is enabled, the LLM analyzes the directory structure to determine which paths contain application code versus runtime/system files.

## Agentic Layout Resolution

When a container image has an ambiguous directory structure, the agentic enrichment layer can resolve which directories contain the actual application code. This is triggered automatically when:

- Agentic mode is enabled (`--llm-model` is set).
- The extracted filesystem has multiple candidate application directories.
- The extractor marks the layout as `needs_agentic=True`.

The agent examines file listings, Dockerfile metadata, and directory naming patterns to select the correct application root.

## Troubleshooting

- **"No container runtime found"** — Install Docker, Podman, or another supported runtime. Alternatively, use `--container-extraction-tier syft` for metadata-only analysis.
- **Extraction timeout** — Large images may take time to extract. The extraction runs synchronously; consider pulling the image first (`docker pull my-app:latest`) to separate network transfer from analysis.
- **Wrong source directory extracted** — Use agentic mode (`--llm-model`) to enable smart layout resolution. Or, extract manually and scan the directory directly.
- **Syft not found** — Syft is optional. Install it for richer SBOM metadata, but scans work without it.
