# vllm-ascend-deepseek-v41-a3

**在单台 8×Ascend 910C（A3，16 die）上原生部署 DeepSeek-V4.1-Flash-1017-w8a8 的完整方案。**
默认不依赖 docker（rootfs 直跑）；也提供 **docker 运行时**，适合离线 / 旧 glibc 宿主
（如 Kylin V10）。rootfs、启动参数、优化补丁、watchdog、压测与排障脚本全部放在本仓库。

> Native, reproducible deployment of **DeepSeek-V4.1-Flash-1017-w8a8** on one
> 8-card Ascend A3 node with vLLM Ascend v0.30.0rc2 — no Docker, just an
> extracted rootfs plus a hardened launcher.

---

## 特性

- **原生 rootfs**：直接从 `quay.nju.edu.cn/ascend/vllm-ascend:v0.30.0rc2-a3` 按 OCI 层解包复用
  镜像内 CANN 9.2.0 / Python 3.12.13 / 预编译算子，host NPU driver 直连，与同机其它 vLLM 环境隔离；
  解包流程（manifest、whiteout、SHA-256、执行位）全部脚本化。离线 / 旧 glibc 宿主可用
  `DSV41_RUNTIME=docker`（见 [`docs/offline-deployment.md`](docs/offline-deployment.md)）。
- **进程安全**：所有服务进程带 `DSV41_PROC_TAG`，watchdog / `STOP=1` / `kill_vllm.sh`
  只清理本部署，不影响同机其它 NPU 任务。
- **生产级参数**：DP2/TP8、Model Runner V2、DSpark K=5、FULL_DECODE_ONLY（37 个 ACL graph 桶）、
  1M 上下文、prefix caching、`prompt_tokens_details`、Engram CPU offload。
- **7 个补丁（6 + 1）+ 6 个优化 gate**：默认值固化进源码树（规避 spawn worker 不继承环境变量），
  含 KV 池 MRV2 抢占修复、tokenizer 控制 token 逃逸修复、多流 vector 核预算（`0202`）。
  详见 [`docs/patches.md`](docs/patches.md)。
- **前缀 / KV 感知路由**：external LB 下 DP0/DP1 独立端口，router 消费 KV 事件做最长前缀亲和，
  `cost` 策略按 token 打分（未命中 prompt + prefill 排队 + KV 饱和罚分）。
  详见 [`docs/router.md`](docs/router.md)。
- **外部 KV 池（opt-in）**：`AscendStoreConnector` + `memcache_hybrid` MetaService + `host_shm`，
  把 prefix cache 扩展到宿主 DRAM（生产 32 GB/rank × 16 = 512 GiB）；不走跨卡 HIXL。
  详见 [`docs/kv-pool.md`](docs/kv-pool.md)。
- **可观测 / 无人值守**：`analyze_serve_log.py` 量化停顿与 lockstep；`watchdog.sh` 健康检查、
  自动重启、日志轮转；`tests/` 含冒烟、路由、补丁回归测试。

## 实测吞吐

单机 8×910C（DP2/TP8），`completions` 接口，约 2K prompt / 256 output，
temperature=0，DSpark 推测解码；**16384 批次 + external LB/router**。

### 2026-11-05 生产实例实测记录

外部 KV 池 `host_shm` 开启，`scripts/bench_2k.py` / `scripts/bench_dual.py`，
每档 3 发中位；压测窗口内 serve 日志无其它在线请求。**这是历史记录，不是当前实例状态。**

| 并发 | 单前缀总吞吐 (tok/s) | 单前缀单流 | 单前缀 TTFT | 双前缀总吞吐 (tok/s) | 双前缀单流 | 双前缀 TTFT |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 87.6 | 98.4 | 308 ms | — | — | — |
| 2 | 186.1 | 104.7 | 295 ms | 177.9 | 107.6 | 479 ms |
| 4 | 340.0 | 95.8 | 305 ms | 333.2 | 100.8 | 515 ms |
| 8 | 645.1 | 91.9 | 336 ms | 729.3 | 108.7 | 379 ms |
| 16 | 1,101.9 | 80.8 | 618 ms | 1,292.9 | 94.2 | 405 ms |
| 32 | 1,834.6 | 73.7 | 667 ms | **1,938.9** | 69.4 | 460 ms |
| 64 | — | — | — | **2,009.8**（持续×12 波） | 54.9 | 671 ms |

单前缀是同一热 prompt；双前缀用 `scripts/bench_dual.py` 让两个长前缀分别落到两个 DP rank，
更接近 DP2 实际并发能力。压测全程 0 失败请求。

### 2026-11-12 优化后实测记录

KV 池同配置开启（16×32 GB）、其余生产参数不变，隔离节点 new_rl：

| 指标 | 生产实例（优化前） | 优化后 | Δ |
|---|---:|---:|---:|
| 单流 tok/s | 100.8 | **107.1** | **+6.2%** |
| 双前缀 c16 tok/s | 1,192 | **1,322** | +10.9% |
| 双前缀 c64 tok/s | 1,854 | 1,872 | +0.9% |
| TTFT（热前缀） | 298 ms | 255–306 ms | 噪声内（无稳定改善） |

按并发展开（优化后，双前缀）：c2 153.2 / c4 280.5 / c8 730.2 / c16 **1,322.0** /
c32 1,678.6 / c64 **1,871.5** tok/s。优化项与完整数据见
[`docs/optimization-deep-dive-2026-11-12.md`](docs/optimization-deep-dive-2026-11-12.md)。

### 8K 复测（2026-11-05）

单前缀 8K→1024 在 C=1/4/8/16 为 105.8 / 411.0 / 842.8 / 1,452.4 tok/s；
双前缀 8K→1024 在 C=2/4/8/16 为 210.7 / 403.9 / 764.8 / **1,392.0** tok/s
（接受率 51.3–51.6%，tokens/step ≈ 3.57）。

### 冷启动与显存

冷启动（权重 + KV + 图捕获）约 5–9 min；权重 ≈38.7 GiB/die。
16384 批次下激活峰值 ≈8.2 GiB、KV ≈6.7 GiB（1,268,538 tokens/DP rank，1M 上下文 1.21×）；
8192 批次 KV ≈9.7 GiB（2,439,086 tokens/rank，2.33×），但因冷 prefill 变慢，
生产保持 16384。历史批次 A/B、8192 取舍与 2026-11-04 复盘见
[`docs/performance.md`](docs/performance.md)。

## 软硬件基线

| 项 | 版本 / 配置 |
|---|---|
| 硬件 | 单机 8×Ascend 910C（A3，每卡 2 die → 16 device，64 GB HBM/die） |
| 架构 / OS | aarch64 / Linux（Ascend 驱动 25.2.1） |
| 镜像 | `quay.nju.edu.cn/ascend/vllm-ascend:v0.30.0rc2-a3` |
| CANN | 9.2.0（镜像内 `/usr/local/Ascend/cann-9.2.0`） |
| Python | 3.12.13（镜像内） |
| vLLM | 0.30.0 |
| vLLM Ascend | 0.30.0rc2（`8fde6d4e3`） |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post7 |
| transformers | 5.14.1 |
| 权重 | `Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8`，W8A8_DYNAMIC，728 GB / 142 shard |
| 上下文 | 1,048,576 tokens |
| 服务名 | `DeepSeek-V4.1-Flash-1017`（必须与客户端 `model` 一致） |

## 目录结构

```
config/        dsv41.env.example（全部可调参数）、dsv41.docker.env.example
scripts/       setup_rootfs.sh  apply_patches.sh  serve.sh  watchdog.sh  kill_vllm.sh
               doctor.sh  prefix_router.py  analyze_serve_log.py
               bench_2k.py  bench_8k.py  bench_dual.py  patches/（补丁权威副本）
docs/          deployment / offline-deployment / architecture / router / performance
               kv-pool / patches / troubleshooting / api
tests/         smoke_test.py + 路由 / 容器 / 日志分析 / 补丁回归测试
systemd/       可选 systemd 单元（含 dsv41-container.service）
examples/      curl_chat.sh、openai_chat.py
```

## 快速开始

> 需要 root（或对 rootfs 目录与 NPU 设备有权限的账号），约 750 GB 权重空间、
> 25 GB 以上空闲空间用于 rootfs。
>
> **离线服务器 / 宿主 glibc < 2.35（如 Kylin V10）**：不要走下面的 rootfs 流程，
> 直接看 [`docs/offline-deployment.md`](docs/offline-deployment.md)（`DSV41_RUNTIME=docker`）。

```bash
git clone <this-repo> vllm-ascend-deepseek-v41-a3
cd vllm-ascend-deepseek-v41-a3

# 1) 配置（至少确认 ROOTFS / MODEL / API KEY）
cp config/dsv41.env.example config/dsv41.local.env
$EDITOR config/dsv41.local.env

# 2) 下载权重（约 728 GB）
pip install modelscope
modelscope download --model Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8 \
  --local_dir models/DeepSeek-V4.1-Flash-1017-w8a8
# 若用仓库内目录，请在 config/dsv41.local.env 设置：
#   DSV41_MODEL=$PWD/models/DeepSeek-V4.1-Flash-1017-w8a8

# 3) 解包 rc2-a3 rootfs（约 21 GB，自动处理 OCI whiteout）
bash scripts/setup_rootfs.sh

# 4) 应用 5 个 vllm-ascend 补丁 + 1 个 vllm 补丁（幂等）
bash scripts/apply_patches.sh apply
bash scripts/apply_patches.sh verify

# 5) 只读预检（硬件 / rootfs / 权重 / 补丁 / 端口）
bash scripts/doctor.sh

# 6) 启动服务（首次约 6 分钟）
#    默认 external 模式：router 8021 + supervisor 8020 + rank 8022/8023
bash scripts/serve.sh
STATUS=1 bash scripts/serve.sh

# 7) 验证
python3 tests/smoke_test.py
curl --noproxy '*' http://127.0.0.1:8021/router/stats \
  -H "Authorization: Bearer $DSV41_API_KEY"
```

生产环境建议由 watchdog 托管：

```bash
bash scripts/watchdog.sh start      # 自动拉起服务并持续监控
bash scripts/watchdog.sh status
bash scripts/watchdog.sh stop       # 只停 watchdog，不影响正在运行的服务
```

停止并清理全部进程：

```bash
bash scripts/kill_vllm.sh
```

## 调用示例

```bash
curl --noproxy '*' http://127.0.0.1:8021/v1/chat/completions \
  -H "Authorization: Bearer $DSV41_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-V4.1-Flash-1017",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 128
  }'
```

模型是 reasoning 模型，原始响应中含 `reasoning`（思考）与 `content`（最终答案）。OpenAI SDK 可直接读取 `message.reasoning`；部分 Anthropic 兼容层会把它映射为 `reasoning_content`。
更多客户端示例见 [`docs/api.md`](docs/api.md)。

## 重要提示

- **鉴权**：`DSV41_API_KEY` 为空时服务不带鉴权，脚本会显式警告；请设置强随机
  key，并仅在可信网络内使用。
- **权重路径**：不要随意改动模型目录。vLLM 编译缓存与模型路径相关，改路径等价于
  一次完整的冷启动。
- **NFS 上的权重**：大模型在 NFS 上并发 mmap 可能触发 page-lock 卡死（worker 进入
  D 状态）。长期运行建议先把权重复制到本地 NVMe。
- **清理范围**：watchdog 重启、`STOP=1 serve.sh`、`kill_vllm.sh` 默认只处理**本部署**
  的进程（serve.sh 启动的进程都带 `DSV41_PROC_TAG`，孤儿 worker 也能识别），同机其它
  vLLM / NPU 任务不受影响。只有显式 `DSV41_CLEAN_ALL_VLLM=1`（watchdog/STOP）或
  `kill_vllm.sh --all` 才会清理本机所有 `VLLM::` worker。
- **NPU 显存**：本服务按 `DSV41_GPU_MEM_UTIL`（默认 0.9）要求每个 die 有足够空闲 HBM。
  同机若还有别的模型占着这些 die，需要先停掉，或调低 `DSV41_GPU_MEM_UTIL`。

## 文档

| 文档 | 内容 |
|---|---|
| [offline-deployment.md](docs/offline-deployment.md) | 离线服务器 + 旧 glibc 宿主：docker 运行时完整部署指南 |
| [deployment.md](docs/deployment.md) | 从零到可服务的完整步骤与校验点（rootfs 直跑） |
| [architecture.md](docs/architecture.md) | rootfs 隔离、DP/TP 进程拓扑、端口与 PID |
| [performance.md](docs/performance.md) | 实测吞吐、显存模型、每个参数的收益与代价 |
| [kv-pool.md](docs/kv-pool.md) | 外部 KV 池（AscendStoreConnector + MemCache）：容量、启用与回滚 |
| [patches.md](docs/patches.md) | 5+1 个补丁、6 个 gate、内容指纹与回归测试 |
| [troubleshooting.md](docs/troubleshooting.md) | 症状 → 原因 → 修复 |
| [api.md](docs/api.md) | OpenAI 兼容接口、SDK、Claude Code |

## 许可证与致谢

本仓库以 **Apache-2.0** 发布。补丁修改的文件来自
[vLLM](https://github.com/vllm-project/vllm) 与
[vLLM Ascend](https://github.com/vllm-project/vllm-ascend)（Apache-2.0）。
模型权重由 Eco-Tech 发布（模型卡片标注 MIT License），本仓库不分发权重。
完整声明见 [`NOTICE`](NOTICE)。
