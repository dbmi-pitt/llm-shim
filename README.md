# llm-shim

OpenAI-compatible API shim for routing chat and embeddings requests to one or more LLM providers via [pydantic-ai](https://ai.pydantic.dev/).

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

Configuration uses a required top-level `providers` map keyed by pydantic-ai
provider IDs. Each provider has separate chat and embeddings model patterns
(exact or wildcard) plus optional per-provider env/settings:

```yaml
providers:
	openai:
		chat_models: ["gpt-*"]
		embedding_models: ["text-embedding-3-*"]
		chat_model_settings: {}
		embedding_model_settings: {}
		env:
			OPENAI_API_KEY: "sk-..."

	bedrock:
		chat_models: ["anthropic.claude*"]
		embedding_models: ["amazon.titan-embed-text-*"]
		env:
			AWS_ACCESS_KEY_ID: "ak-..."
			AWS_SECRET_ACCESS_KEY: "sk-..."
			AWS_DEFAULT_REGION: "us-east-1"

server:
	host: "0.0.0.0"
	port: 8000
```

Routing behavior:

- Chat requests match `providers.<id>.chat_models` in order.
- Embeddings requests match `providers.<id>.embedding_models` in order.
- Exact strings are supported (`"gpt-4o-mini"`) as well as wildcards (`"haiku*"`, `"*"`).
- If `model` is omitted, the first exact (non-wildcard) pattern is used for that route type.

## API examples

### Chat completion

```bash
curl -s http://localhost:8000/v1/chat/completions \
	-H 'Content-Type: application/json' \
	-d '{
		"model": "gpt-4o-mini",
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
		"model": "gpt-4o-mini",
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
		"model": "text-embedding-3-small",
		"input": "llm-shim"
	}'
```
