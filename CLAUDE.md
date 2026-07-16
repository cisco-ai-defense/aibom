# Contributor guidance for AI coding agents

`cisco-ai-defense/aibom` is a **public, Apache-2.0-licensed** project. This file gives AI
coding agents (Claude Code and others) — and their humans — the conventions to follow when
proposing changes here. Everything committed to this repository is world-readable, so keep
it clean of anything internal.

Detailed rules live in [`.claude/rules/`](.claude/rules/) and load automatically alongside
this file.

## Summary

- **Conventional commits** — `feat` / `fix` / `docs` / `chore` / `test`, with an optional
  scope (e.g. `feat(agentic): …`), matching `main`'s history. Reference work by pull-request
  number.
- **Keep the repository public-clean** — no internal-only identifiers of any kind in commit
  messages, code, comments, docstrings, user-facing strings, tests, or PR text. Some PRs
  from Cisco contributors originate from issues or enhancements tracked in Cisco-internal
  systems; keep those internal references out of this repository and link work by PR number
  instead.
- **No AI-attribution trailers** — don't add "Generated with …" lines or `Co-Authored-By`
  trailers for AI tools.
- **Synthetic test fixtures only** — never commit real data, credentials, or live endpoints.
- **Apache-2.0 license headers** on new source files, matching sibling files.
- **Formatting** — the code base is Black-formatted with an 88-column target. There is no
  lint/format gate in CI, so keep new code within the limit by hand and match the
  surrounding style.

See [`.claude/rules/oss-contributions.md`](.claude/rules/oss-contributions.md) for the full
detail, and [`CONTRIBUTING.md`](CONTRIBUTING.md) for the human-facing contribution guide.
