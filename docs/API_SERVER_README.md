# AI BOM UI Mode API Server

The AI BOM tool can start a FastAPI server that serves component data for the React UI. This server runs in-memory and is created when you invoke the main `cisco-aibom` CLI with `--output-format ui`.

## Prerequisites

- Python 3.11+
- uv (Python package manager, recommended)
- cisco-aibom CLI installed

### Install the CLI (if needed)

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: brew install uv

uv tool install --python 3.13 cisco-aibom

# Verify installation
cisco-aibom --help
```

For local development installs, see `README.md` in the repo root.

## Usage

```bash
cd aibom
cisco-aibom analyze /path/to/project --output-format ui
```

There is no separate server-only CLI; use the main `cisco-aibom` entry point for API server mode.

The server listens on `http://127.0.0.1:8000` by default.

## Endpoints

- `GET /api/components` (optional query params: `type`, `file_path`)
- `GET /api/components/types`
- `GET /api/components/{id}`
- `GET /health`

### Example response

```json
{
  "components": [
    {
      "id": "1",
      "name": "langchain_community.llms.openai.OpenAI",
      "file_path": "/app/app.py",
      "line_number": 32,
      "type": "model",
      "text": null,
      "model_name": "gpt-3.5-turbo",
      "embedding_model": null,
      "additional_data": {
        "source": "langchain-app:latest"
      }
    }
  ],
  "total": 1
}
```

## File layout

```
aibom/src/aibom/api/server.py          # FastAPI server
aibom/src/aibom/models/component.py   # Pydantic models
aibom/src/aibom/utils/dataframe_converter.py
aibom/src/aibom/ui_handler.py         # Server startup logic
aibom/src/aibom/cli.py                # --output-format ui entry point
```

## Testing

```bash
cd aibom
uv run pytest tests -v
```

## Notes

- CORS is enabled for all origins to simplify local UI development.
- The server is in-memory only; it does not persist results.
- If no components are found, the UI server exits early (the DataFrame is empty).
