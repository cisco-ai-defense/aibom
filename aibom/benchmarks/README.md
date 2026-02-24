# AIBOM Benchmarks

This directory contains the benchmarking tooling for evaluating the AIBOM analyzer's detection accuracy against real-world AI repositories.

## Running Benchmarks

```bash
cd aibom
uv run python benchmarks/run_benchmarks.py --clone-dir benchmarks/_repos --output benchmarks/results.json
```

This will:

1. **Shallow-clone** 14 repositories from `langchain-ai` and `crewAIInc` into `benchmarks/_repos/`.
2. **Analyze** each repo using the full AIBOM pipeline (Python files, Jupyter notebooks, `langgraph.json` configs).
3. **Write** a consolidated JSON report to `benchmarks/results.json`.

The script takes approximately 5-7 minutes depending on network speed (cloning) and repo sizes.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--clone-dir` | `benchmarks/_repos` | Directory to clone repos into |
| `--output` | `benchmarks/results.json` | Path for the output JSON report |

## Output

`results.json` contains per-repo and aggregate data:

- **Per repo:** file counts, component lists with file paths and line numbers, relationship types.
- **Aggregate:** total components by category (`agent`, `model`, `tool`, `embedding`, `memory`, `datastore`, `prompt`, `other`) and relationships by type (`USES_LLM`, `USES_TOOL`, `USES_MEMORY`, `USES_EMBEDDING`).

## Directory Structure

```
benchmarks/
  run_benchmarks.py   # Benchmark runner script
  results.json        # Output (generated, not committed)
  _repos/             # Cloned repositories (gitignored)
  README.md           # This file
```

## Notes

- The `_repos/` directory is **gitignored**. It is populated on-the-fly by the benchmark script.
- If repos have already been cloned, the script skips re-cloning them.
- To force a fresh clone, delete the `_repos/` directory and re-run.
