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

3. Edit `data/config.yaml` with provider routes only. Provider secrets are
   environment-variable references; do not put key values in YAML.

4. Set the separate inbound-shim and upstream-provider credentials, then start
   the server:

```bash
export LLM_SHIM_API_KEYS='local:replace-with-a-long-random-secret'
export OPENAI_API_KEY='provider-secret-from-your-secret-store'
uv run main.py
```

5. Verify the authenticated API:

```bash
curl -s http://127.0.0.1:8000/v1/models \
  -H 'Authorization: Bearer replace-with-a-long-random-secret'
```

## Quick Start (Docker Compose)

```bash
docker compose up --build
```

`compose.yaml` mounts `./data` to `/config`, and the container reads config from `/config/config.yaml`.
It publishes only `127.0.0.1:8000` by default. Set `LLM_SHIM_API_KEYS` in the
shell or an orchestrator secret, never in the Compose file.

## Security and deployment

Native startup binds `127.0.0.1`. An external bind (including `0.0.0.0`) must
set `security.expose_network: true` and have inbound keys loaded from
`LLM_SHIM_API_KEYS` (`name:key,name2:key`) or a mounted key file. Environment
settings override YAML (for example `LLM_SHIM_SERVER__HOST=127.0.0.1`).
Anonymous access is available only with the explicit
`security.allow_anonymous_loopback: true` development setting.

Every `/v1/*` request uses `Authorization: Bearer` authentication:

```bash
curl -s http://127.0.0.1:8000/v1/models \
  -H 'Authorization: Bearer replace-with-a-long-random-secret'
```

Existing unauthenticated clients must add that header. `/healthz` and `/livez`
are intentionally minimal and unauthenticated; restrict their proxy/load-
balancer reachability separately. Docs/OpenAPI are disabled by default.

For VPN or AWS, terminate TLS at a trusted proxy/ALB, allow only the proxy/ALB
to reach the task or instance security group, and keep shim authentication on.
Run the container with `server.host: 0.0.0.0` only with
`security.expose_network: true` and keys configured; Docker can still publish
only `127.0.0.1:8000`. Do not trust `X-Forwarded-*` headers unless a known
proxy is explicitly configured. Prefer EC2/ECS IAM roles for Bedrock.

## Configuration

Copy and edit [`config.example.yaml`](config.example.yaml); it is the canonical
complete configuration example. Keep it in `data/config.yaml`, which is
excluded from Docker builds and version control.

- `providers` defines route IDs, allowed model patterns, provider settings, and
  TEI endpoints.
- `environment` contains non-secret literal provider values, such as
  `AWS_DEFAULT_REGION: us-west-2`.
- `env` maps a provider variable to the name of a runtime variable that holds
  its secret, for example `OPENAI_API_KEY: OPENAI_API_KEY`.
- `security` controls incoming-client authentication and resource limits.
- `server.host` defaults to `127.0.0.1`. Use `0.0.0.0` only for a deliberately
  exposed container/proxy deployment with `security.expose_network: true`.

For Bedrock, prefer an EC2/ECS IAM role and retain only the non-secret region:

```yaml
bedrock:
  chat_models: ["us.anthropic.claude-*", "us.anthropic.claude-haiku-*"]
  embedding_models: ["amazon.titan-embed-text-*"]
  embedding_model_settings:
    dimensions: 512
  environment:
    AWS_DEFAULT_REGION: "us-west-2"
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
  -H 'Authorization: Bearer replace-with-a-long-random-secret' \
  -d '{
    "model": "openai:gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello, world!"}]
  }'
```

```powershell
$body = @{
    model = "openai:gpt-4o-mini"
    messages = @(
        @{
            role = "user"
            content = "Hello, world!"
        }
    )
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
    -Uri "http://localhost:8000/v1/chat/completions" `
    -Method Post `
    -ContentType "application/json" `
    -Headers @{ Authorization = "Bearer replace-with-a-long-random-secret" } `
    -Body $body
```

Embeddings:

```bash
curl -s http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer replace-with-a-long-random-secret' \
  -d '{
    "model": "openai:text-embedding-3-small",
    "input": "Hello, world!"
  }'
```

```powershell
$body = @{
    model = "openai:text-embedding-3-small"
    input = "Hello, world!"
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
    -Uri "http://localhost:8000/v1/embeddings" `
    -Method Post `
    -ContentType "application/json" `
    -Headers @{ Authorization = "Bearer replace-with-a-long-random-secret" } `
    -Body $body
```

TEI-backed embeddings through the shim:

```bash
curl -s http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer replace-with-a-long-random-secret' \
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
curl -s http://localhost:8000/v1/models \
  -H 'Authorization: Bearer replace-with-a-long-random-secret'
```

## External References

- OpenAI API format: [Chat Completions](https://platform.openai.com/docs/api-reference/chat/create), [Embeddings](https://platform.openai.com/docs/api-reference/embeddings/create), [Models](https://platform.openai.com/docs/api-reference/models/list)
- pydantic-ai providers and model naming: [docs](https://ai.pydantic.dev/models/)

This project is a thin shim that routes OpenAI-compatible requests to pydantic-ai providers. Those docs will serve you better on how to make requests and configure providers.
