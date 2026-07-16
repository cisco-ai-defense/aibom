---
description: Conventions for contributing to the public cisco-ai-defense/aibom repository
---

# Contributing to this public repository

`cisco-ai-defense/aibom` is a **public, Apache-2.0-licensed** project. Everything committed
here is world-readable. Apply these rules to every commit, code change, test, and pull
request.

## Commit messages

- Use **Conventional Commits**: `feat`, `fix`, `docs`, `chore`, `test`, optionally scoped
  (e.g. `feat(agentic): …`), matching the existing `main` history.
- Reference work by **pull-request number**. Do not reference internal issue trackers.
- Do **not** add AI-attribution trailers such as `Co-Authored-By: <AI tool>` or
  "Generated with …" lines.

## Keep the repository public-clean

Because this repository is public, never include internal-only identifiers or details
anywhere they would ship — commit messages, code, comments, docstrings, user-facing strings
(CLI help, log messages), test names, or pull-request titles and descriptions. This includes:

- **Internal issue identifiers and tracking systems.** Some PRs from Cisco contributors
  originate from issues or enhancements tracked in Cisco-internal systems. Keep those
  internal references out of the repository and track intent by PR number instead.
- **Internal infrastructure details** — cloud project or account identifiers, private
  endpoints or deployment names, internal hostnames.
- **Secrets** — credentials, API keys, or tokens of any kind.
- **Internal-only anecdotes or figures** — generalize them if they add value publicly.

The scrub applies to **code, comments, docstrings, user-facing strings, and tests** — not
just commit messages. Before committing, review the diff *and* the message for anything in
the list above.

## Tests

- Use **synthetic fixtures only**. Never commit real data, credentials, or live endpoints.

## Licensing

- New source files carry the **Apache-2.0 license header**, matching the sibling files in
  the same directory.

## Formatting

- The code base is **Black-formatted** with an 88-column target. There is no lint or format
  gate in CI, so keep new and modified code lines within the limit by hand and match the
  surrounding style.
