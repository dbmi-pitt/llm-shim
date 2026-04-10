# llm-shim

`llm-shim` exposes an OpenAI-compatible API and routes requests to configured providers via [pydantic-ai](https://ai.pydantic.dev/).

## Endpoints

- `POST /v1/chat/completions`
- `POST /v1/embeddings`
- `GET /v1/models`
- `GET /healthz` and `GET /livez`

## Requirements

- Python `>=3.14`
- [uv](https://docs.astral.sh/uv/) (recommended)

## Quick Start (Local)

1. Install dependencies:

```bash
uv sync
```

2. Create config file in `data/`:

```bash
cp config.example.yaml data/config.yaml
```

3. Edit `data/config.yaml` with your providers and credentials.

4. Start the server:

```bash
uv run main.py
```

5. Open:

- `http://localhost:8000/docs`
- `http://localhost:8000/redoc`

## Quick Start (Docker Compose)

```bash
docker compose up --build
```

`compose.yaml` mounts `./data` to `/config`, and the container reads config from `/config/config.yaml`.

## Configuration

- `providers`: a mapping of [provider names](https://ai.pydantic.dev/models/overview/) to their configuration.
  - `chat_models`: list of allowed chat model patterns (supports wildcards, example: `gpt-*`).
  - `embedding_models`: list of allowed embedding model patterns (supports wildcards, example: `text-embedding-3-*`).
  - `backend`: optional embedding backend selector. Defaults to `pydantic_ai`; use `tei` for a HuggingFace Text Embeddings Inference endpoint.
  - `chat_model_settings`: optional kwargs to pass to chat model calls. See [pydantic-ai docs](https://ai.pydantic.dev/agent/#model-run-settings) for supported settings.
  - `embedding_model_settings`: optional kwargs to pass to embedding model calls. See [pydantic-ai docs](https://ai.pydantic.dev/embeddings/) for supported settings.
  - `env`: environment variables to set for this provider (required for things like API keys).
  - `tei`: required when `backend: tei`.
    - `base_url`: TEI server base URL.
    - `endpoint`: TEI embeddings path. Defaults to `/embed`.
    - `auth_token_env`: optional env var name whose value is sent as `Authorization: Bearer ...`.
    - `request_timeout_seconds`: HTTP timeout for the TEI request.
    - `input_prefix_template`: optional template used to transform each input before sending it upstream. Use `{input}` as the placeholder.
- `server`: FastAPI server settings.
  - `host`: server host (default: `"0.0.0.0"`)
  - `port`: server port (default: 8000)
  - `reload`: enable auto-reload (default: `false`)
  - `workers`: number of worker processes (default: `1`)
  - `log_level`: logging level (default: `"info"`)

```yaml
providers:
  openai:
    chat_models: ["gpt-*"]
    embedding_models: ["text-embedding-3-*"]
    chat_model_settings: {}
    embedding_model_settings: {}
    env:
      OPENAI_API_KEY: "sk-..."

  google-gla:
    chat_models: ["gemini-*"]
    embedding_models: ["gemini-embedding-*"]
    env:
      GOOGLE_API_KEY: "AIza..."

  tei-qwen:
    backend: tei
    embedding_models: ["Qwen/Qwen3-Embedding-0.6B"]
    env:
      TEI_API_TOKEN: "optional-bearer-token"
    tei:
      base_url: "http://embedding-box:8080"
      endpoint: "/embed"
      auth_token_env: "TEI_API_TOKEN"
      request_timeout_seconds: 30
      input_prefix_template: "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {input}"

server:
  host: "0.0.0.0"
  port: 8000
  reload: false
  workers: 1
  log_level: "info"
```

### Routing Rules

- Requests must send model as `provider:model` (example: `openai:gpt-4o-mini`).
- The `provider` must exist in `providers`.
- Chat models are validated against `chat_models`, and embeddings against `embedding_models`.
- Model patterns support exact values and wildcards (`*`) using shell-style matching.
- `backend: tei` providers support embeddings only and call the configured TEI HTTP endpoint directly.

## Minimal API Examples

Chat completion:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openai:gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello, world!"}]
  }'
```

Embeddings:

```bash
curl -s http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openai:text-embedding-3-small",
    "input": "Hello, world!"
  }'
```

TEI-backed embeddings through the shim:

```bash
curl -s http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tei-qwen:Qwen/Qwen3-Embedding-0.6B",
    "input": [
      "What is the capital of China?",
      "Explain gravity"
    ]
  }'
```

List configured models:

```bash
curl -s http://localhost:8000/v1/models
```

## External References

- OpenAI API format: [Chat Completions](https://platform.openai.com/docs/api-reference/chat/create), [Embeddings](https://platform.openai.com/docs/api-reference/embeddings/create), [Models](https://platform.openai.com/docs/api-reference/models/list)
- pydantic-ai providers and model naming: [docs](https://ai.pydantic.dev/models/)

This project is a thin shim that routes OpenAI-compatible requests to pydantic-ai providers. Those docs will serve you better on how to make requests and configure providers.
