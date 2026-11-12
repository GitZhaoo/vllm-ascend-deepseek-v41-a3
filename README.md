# vllm-ascend-deepseek-v41-a3

**在单台 8×Ascend 910C（A3，16 die）上原生部署 DeepSeek-V4.1-Flash-1017-w8a8（W8A8）的完整方案。**
默认不依赖 docker（rootfs 直跑），也提供 docker 运行时（离线 / 旧 glibc 宿主）。
rootfs、启动参数、优化补丁、watchdog、压测与排障脚本全部放在本仓库。

> Native, reproducible deployment of **DeepSeek-V4.1-Flash-1017-w8a8** on one
> 8-card Ascend A3 node with vLLM Ascend v0.30.0rc2 — no Docker, just an
> extracted rootfs plus a hardened launcher.

---

## 特性

- **原生 rootfs**：直接从 `quay.nju.edu.cn/ascend/vllm-ascend:v0.30.0rc2-a3` 按 OCI 层解包，
  复用镜像内 CANN 9.2.0 / Python 3.12.13 / 预编译算子，host NPU driver 直连；
  解包（manifest / whiteout / SHA-256 / 执行位）全部脚本化；离线或旧 glibc 宿主可用
  `DSV41_RUNTIME=docker`。
- **进程安全**：服务进程带 `DSV41_PROC_TAG`，watchdog / `STOP=1` / `kill_vllm.sh`
  只清理本部署，不影响同机其它 NPU 任务。
- **生产级参数**：DP2/TP8、Model Runner V2、DSpark K=5、FULL_DECODE_ONLY（37 个 ACL graph 桶）、
  1M 上下文、prefix caching、`prompt_tokens_details`、Engram CPU offload。
- **7 个补丁（6 + 1）+ 6 个优化 gate**：默认值固化进源码树（规避 spawn worker 不继承环境变量），
  含 KV 池 MRV2 抢占修复、tokenizer 控制 token 逃逸修复、多流 vector 核预算、KV 池 lookup 快路径。
- **前缀 / KV 感知路由**：external LB 下 DP0/DP1 独立端口，router 消费 KV 事件做最长前缀亲和；
  `cost` 策略按 token 打分（未命中 prompt + prefill 排队 + KV 饱和罚分）。
- **外部 KV 池（默认启用）**：`AscendStoreConnector` + `memcache_hybrid` MetaService + `host_shm`，
  prefix cache 扩展到宿主 DRAM（32 GB/rank × 16 = 512 GiB），不走跨卡 HIXL。
- **可观测 / 无人值守**：`analyze_serve_log.py` 量化停顿与 lockstep；`watchdog.sh` 健康检查 /
  自动重启 / 日志轮转；`tests/` 含冒烟、路由、补丁回归测试。

## 实测性能（当前配置）

口径：单机 8×910C、**W8A8 权重**、DP2/TP8（EP16）、`BAT 16384 / max_num_seqs 32 / 图桶≤192`、
external LB + prefix router、**外部 KV 池（16×32 GB）开启**、DSpark K=5、
`completions` 接口、温度 0、约 2K prompt / 256 输出、每档 3 发取中位。
同配置多次运行方差约 ±5%，下表为当前配置的典型值。

### 单流

| 指标 | 数值 |
|---|---:|
| 单流 decode | **105 tok/s**（104–106） |
| ms/step | **40.5–41.2** |
| TTFT（热前缀） | 255–306 ms |
| TTFT（冷前缀） | 282–337 ms |
| 冷前缀 decode | 114–116 tok/s |

### 并发（双前缀 A/B）

| 并发 | 聚合 tok/s | 单流 tok/s | TTFT |
|---:|---:|---:|---:|
| 2 | 185–200 | 110–115 | ~293 ms |
| 4 | 374–377 | 109–111 | ~338 ms |
| 8 | 726–758 | 111 | 338–460 ms |
| 16 | 1,228–1,352 | 92 | 473–477 ms |
| 32 | 1,891–1,977 | 71–72 | 475–561 ms |
| 64 | 1,913–2,055 | 53–57 | 634–906 ms |

### 与优化前（生产实例）对比

| 指标 | 优化前 | 当前配置 | 变化 |
|---|---:|---:|---:|
| 单流 tok/s | 100.8 | **104–106** | **+4~5%** |
| ms/step | 41.2 | **40.5–41.2** | −1.7%~0 |
| 双前缀 c16 | 1,192 | 1,228–1,352 | +3~13% |
| 双前缀 c64 | 1,854 | 1,913–2,055 | +3~11% |
| TTFT（热前缀） | 298 ms | 255–306 ms | 噪声内 |

容量与启动：权重 ≈38.7 GiB/die；16384 批次下激活峰值 ≈8.2 GiB、KV ≈6.7 GiB
（1,268,538 tokens/DP rank，1M 上下文 1.21×）；冷启动（权重 + KV + 图捕获）约 5–9 min。

## 关键优化（均已入库）

| 项 | 开关默认 | 效果 |
|---|---|---|
| `0202` 多流 vector 核预算（limit_core_num） | **开**（`DSV41_LIMIT_CORE_MAIN_AIV=16 / AUX_AIV=32 / AIC=24`，0 回退） | 单流 +1~4%，ms/step −2%，精度零风险 |
| `0304` KV 池 lookup fast-path | **开**（`VLLM_ASCEND_V41_KVPOOL_SKIP_LOCAL_LOOKUP=1`、`..._BATCH_LOOKUP=1`） | 本地已覆盖时跳过同步 ZMQ + 13 组 exists 合并为 1 次；语义等价，TTFT 收益在噪声内 |
| `0301/0302/0303` 融合补丁 | 关（`VLLM_ASCEND_V41_QNORM_FUSE` 等，实验性） | 配对 ms/step −0.73%（0302/0303 零风险，0301 需精度门） |
| `multistream_engram_overlap` | 关（实验性） | 相对 fusion 再 −0.4~0.6 ms/step，需跨流精度回归 |

## 已排除（不再投入）

- **MegaMoe**（`enable_fused_mc2=2`）：形状白名单拒绝 V4.1-Flash（`moe_intermediate_size=2304`），
  引擎静默回滚；该开关只会关掉 `multistream_overlap_shared_expert` ⇒ decode −5%；
- PGO libpython、KV 池开/关、BAT 8192：实测噪声内或无收益；
- `npugraph_ex` / `static_kernel` / `fused_mc2=1` / `prefill_mc2` / `FULL_AND_PIECEWISE`：
  本镜像启动失败、输出退化或并发回退；
- 扩 `max_num_seqs`（64/128）+ 图桶：图捕获阶段卡死（DP 同步 hang）；
- TP16/DP1（Engram 限定 TP≤8）、W4A8 拓扑、单节点 PD 分离（W8A8 权重放不下）。
- 单流再提升需 kernel 级深度融合（Triton/AscendC，预计 12–24%，按周计）。

## 软硬件基线

| 项 | 版本 / 配置 |
|---|---|
| 硬件 | 单机 8×Ascend 910C（A3，每卡 2 die → 16 device，64 GB HBM/die） |
| 架构 / OS | aarch64 / Linux（Ascend 驱动 25.2.1） |
| 镜像 | `quay.nju.edu.cn/ascend/vllm-ascend:v0.30.0rc2-a3` |
| CANN / Python | 9.2.0 / 3.12.13（镜像内） |
| vLLM / vLLM Ascend | 0.30.0 / 0.30.0rc2（`8fde6d4e3`） |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post7 |
| 权重 | `Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8`（W8A8_DYNAMIC，728 GB / 142 shard） |
| 上下文 | 1,048,576 tokens |
| 服务名 | `DeepSeek-V4.1-Flash-1017`（须与客户端 `model` 一致） |

## 目录结构

```
config/        dsv41.env.example（全部可调参数）、dsv41.docker.env.example
scripts/       setup_rootfs.sh  apply_patches.sh  serve.sh  watchdog.sh  kill_vllm.sh
               doctor.sh  prefix_router.py  analyze_serve_log.py  verify_rootfs.py
               bench_2k.py  bench_8k.py  bench_dual.py  patches/
docs/          deployment / offline-deployment / architecture / router / performance
               kv-pool / patches / troubleshooting / api
tests/         smoke_test.py + 路由 / 容器 / 日志分析 / 补丁回归测试
systemd/       可选 systemd 单元（含 dsv41-container.service）
examples/      curl_chat.sh、openai_chat.py
```

## 快速开始

> 需要 root（或对 rootfs 目录与 NPU 设备有权限的账号）、约 750 GB 权重空间、25 GB 以上空闲空间。
> 离线服务器 / host glibc < 2.35（如 Kylin V10）请直接看
> [`docs/offline-deployment.md`](docs/offline-deployment.md)（`DSV41_RUNTIME=docker`）。

```bash
git clone <this-repo> vllm-ascend-deepseek-v41-a3
cd vllm-ascend-deepseek-v41-a3

# 1) 配置（至少确认 ROOTFS / MODEL / API KEY）
cp config/dsv41.env.example config/dsv41.local.env
$EDITOR config/dsv41.local.env

# 2) 权重（约 728 GB）/ rootfs（约 21 GB）
pip install modelscope
modelscope download --model Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8 \
  --local_dir models/DeepSeek-V4.1-Flash-1017-w8a8
bash scripts/setup_rootfs.sh

# 3) 补丁 + 预检
bash scripts/apply_patches.sh apply && bash scripts/apply_patches.sh verify
bash scripts/verify_rootfs.py manifest "$DSV41_ROOTFS" -o /tmp/ref.tsv   # 可选：rootfs 完整性
bash scripts/doctor.sh

# 4) 启动（首次约 5–9 分钟；external 模式 router 8021 / rank 8022-8023）
bash scripts/serve.sh
STATUS=1 bash scripts/serve.sh
python3 tests/smoke_test.py

# 5) 压测（单前缀 / 双前缀 DP2 并发）
python3 scripts/bench_2k.py  --base-url http://127.0.0.1:8021 --api-key "$DSV41_API_KEY"
python3 scripts/bench_dual.py --base-url http://127.0.0.1:8021 --api-key "$DSV41_API_KEY" \
  --concurrency 2,4,8,16,32,64
```

生产建议由 watchdog 托管：`bash scripts/watchdog.sh start`（`stop` 只停 watchdog，不停服务）；
彻底清理用 `bash scripts/kill_vllm.sh`。

## 文档

| 文档 | 内容 |
|---|---|
| [deployment.md](docs/deployment.md) | 从零到可服务的完整步骤与校验点（rootfs 直跑） |
| [offline-deployment.md](docs/offline-deployment.md) | 离线 / 旧 glibc 宿主的 docker 运行时指南 |
| [architecture.md](docs/architecture.md) | rootfs 隔离、DP/TP 进程拓扑、端口与 PID |
| [performance.md](docs/performance.md) | 参数取舍、显存模型、历史批次 A/B 与复盘 |
| [kv-pool.md](docs/kv-pool.md) | 外部 KV 池：容量、启用与回滚 |
| [patches.md](docs/patches.md) | 补丁与 gate 详解（含 0202 / 0304 的实测） |
| [troubleshooting.md](docs/troubleshooting.md) | 症状 → 原因 → 修复 |
| [api.md](docs/api.md) | OpenAI 兼容接口、SDK、Claude Code 接入 |
| [optimization-deep-dive-2026-11-12.md](docs/optimization-deep-dive-2026-11-12.md) | 本轮优化验证全过程与数据 |

## 许可证与致谢

本仓库以 **Apache-2.0** 发布。补丁修改的文件来自
[vLLM](https://github.com/vllm-project/vllm) 与
[vLLM Ascend](https://github.com/vllm-project/vllm-ascend)（Apache-2.0）。
模型权重由 Eco-Tech 发布（模型卡片标注 MIT License），本仓库不分发权重。
完整声明见 [`NOTICE`](NOTICE)。
