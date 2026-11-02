# Security

## Reporting

If you find a vulnerability in this deployment tooling, please **do not open a
public issue**. Contact the maintainers privately with:

- affected script/version,
- reproduction steps,
- impact assessment.

For vulnerabilities in vLLM, vLLM Ascend, CANN or the model itself, report to
the respective upstream projects.

## Deployment model

This project assumes a **trusted operator on a dedicated inference node**. It
provides no tenant isolation, no request accounting and no rate limiting.

## API access

- Set `DSV41_API_KEY` to a long random value (for example `openssl rand -hex 32`).
  With an empty key the server
  accepts unauthenticated requests; `serve.sh` prints a warning but does not
  block startup for loopback-only testing.
- The server speaks plain HTTP and bearer tokens are sent in clear text.
  Use it only on trusted links and trusted networks.
- Do not put the API key in shell history, committed config or logs. Prefer an
  environment file with `0600` permissions or a secret manager.

## Resource exhaustion

The service has fixed limits (`--max-num-seqs`, `--max-model-len`,
`--max-num-batched-tokens`). A client can still occupy all slots for a long
time with large generations. Add quota/rate limiting when the endpoint is
shared.

## Process and filesystem privileges

Scripts run as root to access NPU devices and the rootfs. Consequences:

- `scripts/kill_vllm.sh` terminates vLLM processes on the whole node by default.
  Set `DSV41_CLEAN_ALL_VLLM=0` when sharing the node.
- Config files are sourced as shell. Only edit files you trust; do not point
  `DSV41_PROJECT_ROOT/config` at world-writable locations.
- The rootfs is extracted from a remote registry. Layer content is verified
  against the manifest's SHA-256 digests, but you must trust the registry and
  the pinned image reference.

## Prompt and output data

vLLM logs metadata by default. Avoid enabling verbose request logging on
production nodes handling sensitive prompts. `/metrics` exposes counters, not
request content.
