# llm-shim

OpenAI-compatible API shim for routing chat and embeddings requests to one or more LLM providers via [Instructor](https://python.useinstructor.com/).

## Endpoints

- `POST /v1/chat/completions`
- `POST /v1/embeddings`

## Requirements

- Python `>=3.14`
- [uv](https://docs.astral.sh/uv/) for dependency and environment management
- API keys for any providers you configure

## Quick start

1. Install dependencies:

```bash
uv sync
```

2. Create local environment config:

```bash
cp config.example.yaml config.yaml
```

1. Edit `config.yaml` with your model and provider details.

2. Start server:

```bash
uv run main.py
```

5. Open docs:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuration

WIP

## Model routing rules

Incoming `model` is resolved in this order:

1. If omitted: use `LLM_SHIM_DEFAULT_PROVIDER`
2. If it matches a configured provider alias: use that provider
3. If it exactly matches a configured provider model string: use that provider
4. Otherwise: return `400`

## API examples

### Chat completion

```bash
curl -s http://localhost:8000/v1/chat/completions \
	-H 'Content-Type: application/json' \
	-d '{
		"model": "default",
		"messages": [
			{"role": "user", "content": "Testing..."}
		]
	}'
```

### Chat completion with JSON schema output

```bash
curl -s http://localhost:8000/v1/chat/completions \
	-H 'Content-Type: application/json' \
	-d '{
		"model": "default",
		"messages": [
			{"role": "user", "content": "Return a short summary and confidence."}
		],
		"response_format": {
			"type": "json_schema",
			"json_schema": {
				"name": "summary_response",
				"schema": {
					"type": "object",
					"properties": {
						"summary": {"type": "string"},
						"confidence": {"type": "number"}
					},
					"required": ["summary", "confidence"]
				}
			}
		}
	}'
```

`content` is returned as a JSON string for schema-based responses.

### Embeddings

```bash
curl -s http://localhost:8000/v1/embeddings \
	-H 'Content-Type: application/json' \
	-d '{
		"model": "default",
		"input": "llm-shim"
	}'
```
