# Changelog

All notable changes to this deployment project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] — 2026-11-04

### Added

- **`cost` routing policy（默认）**：按 token 给 rank 打分（未命中 prompt + 排队 prefill +
  KV 饱和罚分），周期性抓取 `/metrics` 的 `vllm:kv_cache_usage_perc` /
  `vllm:num_requests_waiting`；`DSV41_ROUTER_POLICY=legacy` 可回退到请求数策略。
  新增 `DSV41_ROUTER_KV_HIGH|SAT_PENALTY|PREFILL_WEIGHT|LOAD_INTERVAL`。
- **外部 KV 池（默认开启）**：AscendStoreConnector + Mooncake，把 prefix cache
  扩展到宿主内存（默认 64GB/卡；16 die ≈ 1 TiB ≈ 11.6M tokens/卡，约显存 KV 的 9 倍）。
  跨请求 / 跨 DP rank 复用长文档与多轮前缀，减少重复 prefill（也就减少 DP lockstep
  停顿）。`scripts/kvpool.sh` 管理 master 生命周期；`DSV41_KV_POOL=0` 一键回滚。
  另固定 `PYTHONHASHSEED=0` 保证缓存 key 跨 rank 一致。详见 `docs/kv-pool.md`。
- `/router/stats` 新增 `policy`、`inflight_tokens`、`pending_prefill`、`kv_usage`、
  `waiting`、`spills`、`sat_avoid`、`digest_evicted`、`digest_resets`。
- `scripts/analyze_serve_log.py`（`make analyze LOG=...`）：只读 serve 日志分析，
  量化 DP lockstep 停顿、KV 饱和与前缀命中。
- `tests/test_router_proxy.py`、`tests/test_analyze_serve_log.py`，router 逻辑测试扩充。
- Docker 运行时 / 离线部署：`scripts/container.sh`、`config/dsv41.docker.env.example`、
  `systemd/dsv41-container.service`、`docs/offline-deployment.md`、`tests/test_container.py`。
- 并发压力复测（2026-11-04，README）：双前缀 C=64 持续 ~3.18k tok/s，全程 0 失败。
- Router 重启后 `cost` 策略 A/B（2026-11-04，README）：双前缀 C=32/64 持续吞吐
  +4.8%/+6.0%，TTFT -41%/-45%；单前缀 C=8 +41%；当前为默认策略，
  `DSV41_ROUTER_POLICY=legacy` 可回退（legacy C=64 实测 ~3.17k tok/s）。

### Changed

- 默认开启 `--enable-prompt-tokens-details`。
- 生产批次当晚实验并**回退**：8192 冷 prefill 场景吞吐 -25~-47%（双前缀），KV 容量
  却翻倍到 2,439,086 tokens/rank / 2.33× 并发；最终保持 16384，完整 A/B 表入 README，
  热前缀业务可显式切 8192。
- 发现"引擎 epoch"异常（当晚已定位，见 Fixed）：与 11-01 老引擎配置逐项相同的
  冷重启引擎，混合负载吞吐下降 ~2×。
- 清理范围默认只针对本部署（`DSV41_PROC_TAG`），`DSV41_CLEAN_ALL_VLLM` 默认改为 0。

### Fixed

- **"引擎 epoch 异常"根因定位并修复（当晚）**，实为两个独立问题叠加：
  1. *router `cost` 策略缺 max-skew 均衡护栏*（主因 ~1.9×）：温 digest 表把双前缀
     流量按亲和贴到单一 rank（实测 active 44 vs 20，超出 `max_num_seqs=32` 排队），
     直连 rank 对照（3,430 合计）证明引擎无损；`_choose_cost` 现在与 legacy 共用
     `--max-skew` 护栏，修复后温 digest 态 12 波 C=64 稳定 3,439–3,465 tok/s
     （当晚最佳 3,499）。
  2. *torch/triton 编译缓存被 8192 实验污染*（次因 ~1.6×）：被 8192 冷启动改写的
     共享 inductor/triton 摘要被 16384 引擎半兼容复用（该引擎启动零缓存写入、
     decode 正常混合 prefill 慢）；清空 `~/.cache/vllm/torch_compile_cache` 与
     `~/.triton/cache` 后由 ~1.05K 恢复 ~1.77K。A/B 改批次前应备份这两个目录。
- Router 在决策时**同步预占 active**，并发请求不再全部落到 rank0。
- digest 表上限改为按存活条目计算，只淘汰最旧 10%，不再整表清空。
- `/tokenize` 在健康 rank 间轮询并可失败切换；stale keep-alive 先在同一个 rank 重试；
  响应开始后的失败不再重放到另一个 rank；全部 rank unhealthy 时不再 500。

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
