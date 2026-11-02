# Changelog

All notable changes to this deployment project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.1] — 2026-11-01

### Fixed

- Regenerate `0002-dsa-v41.patch` from the runtime-verified rootfs diff; the
  previously hand-edited revision did not apply to a pristine rc2 tree.
- Detect partially applied patch series instead of treating `envs.py` alone as
  proof that every patch is applied. Patch state now combines per-patch markers,
  forward/reverse `git apply --check` and a gate-normalised content heuristic.
- Pin the Engram WKV gate in `model.py` as well as `engram/layer.py`; both read
  the same switch and a mismatch breaks weight sharding.
- Refuse to start a second instance on an occupied port unless explicitly
  allowed.
- Auto-detect the rootfs Python and CANN directories while keeping override
  variables.
- `serve.sh run` now preserves the Python PID and signal handling.
- Watchdog: detect duplicate/stale instances, fail fast on deterministic startup
  failures, and avoid killing a server that is still inside its cold-start
  window when attached manually.
- `kill_vllm.sh`: add `--port`, accept `ss` or `netstat`.
- Benchmarks: fix completion-token double counting in the 8K script.

### Added

- `scripts/doctor.sh` read-only preflight.
- `scripts/patch_utils.py` and offline unit tests for patch state and OCI
  extraction.
- GitHub Actions CI, `.editorconfig`, `make test`, `make doctor`.
- Setup skips extraction only when the recorded image provenance matches;
  Python site-packages `.so` execute bits are restored too.
- Troubleshooting notes for two benign rc2 warnings (missing KDA Triton
  `causal_conv1d_update_npu` and CUDA-only `vllm._deepselect_C`).

## [0.1.0] — 2026-11-01

### Added

- Reproducible, Docker-free rootfs extraction from
  `quay.nju.edu.cn/ascend/vllm-ascend:v0.30.0rc2-a3` (linux/arm64), including
  OCI whiteout handling, SHA-256 verification, resumable layers and execute-bit
  restoration.
- Production launcher for DeepSeek-V4.1-Flash-1017-w8a8 with DP2/TP8 expert
  parallel, Model Runner V2, DSpark speculative decoding (K=5),
  `FULL_DECODE_ONLY` with 37 graph buckets, 1M context, prefix caching and
  Engram CPU offload.
- Patch series (4 patches) and six optimization gates pinned to default `1` so
  spawn workers actually enable them.
- Health-check watchdog with grace period, restart and log rotation.
- Thorough cleanup script for vLLM processes and NPU worker residue.
- 2K throughput and 8K speculative-decoding benchmarks.
- Smoke test, OpenAI/curl examples, optional systemd units.
- Deployment, architecture, performance, patch and troubleshooting docs.

### Notes

- Reference throughput on one 8×Ascend 910C node: 69.8 tok/s single stream,
  1307.9 tok/s aggregate at concurrency 32 (2K prompt / 256 output).

[0.1.1]: https://example.invalid/vllm-ascend-deepseek-v41-a3/releases/tag/v0.1.1
[0.1.0]: https://example.invalid/vllm-ascend-deepseek-v41-a3/releases/tag/v0.1.0
