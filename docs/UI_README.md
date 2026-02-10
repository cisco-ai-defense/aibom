# UI for AI BOM

<img width="1504" height="1140" alt="Screenshot 2025-08-25 at 6 20 37 PM" src="https://github.com/user-attachments/assets/1a053bca-5638-400c-ad4a-07cfbd348017" />

## Overview

The UI is a Vite + React application that visualizes component analysis results. It can connect to the analyzer UI mode (FastAPI server) or use mocked data in development.

## Stack

- Vite + React (TypeScript)
- Tailwind CSS + shadcn/ui
- Axios + TanStack Query
- MSW for mock API responses in development

## Requirements

- Node.js >= 22
- Python 3.11+ (for the analyzer CLI)
- uv (Python package manager, recommended)
- cisco-aibom CLI installed

## Install the analyzer CLI (if needed)

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: brew install uv

uv tool install --python 3.13 cisco-aibom

# Verify installation
cisco-aibom --help
```

## Run locally

```bash
cd ui
npm ci
npm run dev
```

The dev server runs on `http://localhost:8080` by default (see `ui/vite.config.ts`).

## Connect to the AI BOM UI server

1. Start the AI BOM UI server (via the main `cisco-aibom` CLI):

```bash
cd aibom
cisco-aibom analyze /path/to/project --output-format ui
```

2. Create `ui/.env.local` and point it at the API server:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

3. Start the UI with `npm run dev`.

## Mock API (development)

If `VITE_API_BASE_URL` is not set, MSW intercepts requests to `/api/*` and serves mock data.

- Handlers: `ui/src/mocks/handlers.ts`
- Sample data: `ui/src/mocks/sampleData.ts`
- MSW bootstrap: `ui/src/main.tsx`

If `VITE_API_BASE_URL` is set, the UI makes real network requests and MSW will not intercept cross-origin calls.

## Key UI features

- Component table with search, filter, and selection
- Summary statistics by component type
- Export JSON for filtered components

## API expectations

The UI expects the AI BOM API endpoints below:

- `GET /api/components`
- `GET /api/components/types`
- `GET /api/components/{id}`
- `GET /health`
