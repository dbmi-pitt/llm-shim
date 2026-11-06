# Security policy and deployment boundary

`llm-shim` is an authenticated gateway to configured upstream model credentials.
Possession of a shim bearer key permits use of every model that the deployment
has configured; use separate deployments/keys when tenants need isolation.

Production checklist:

- Set `security.expose_network: true`, configure `LLM_SHIM_API_KEYS` or a
  mounted `security.api_keys_file`, and use TLS at a trusted reverse proxy.
- Keep Docker port publishing private (or omit it behind a proxy); use security
  groups/firewalls as an independent control. An ALB alone is not authentication.
- Disable reload and docs, rotate named keys, and use a secret manager.
- On EC2/ECS use IAM roles for Bedrock, not static AWS access keys.

The built-in limiter and concurrency bound are per process. Workers/replicas
need a shared proxy or gateway rate limiter. Provider credential overrides use
a process-global compatibility mechanism and are serialized in one process;
they are not a security boundary against unrelated code in that process.
