# vllm-ascend-deepseek-v41-a3

**在单台 8×Ascend 910C（A3，16 die）上原生部署 DeepSeek-V4.1-Flash-1017-w8a8 的完整方案。**
默认不依赖 docker（rootfs 直跑）；也提供 **docker 运行时**，适合离线 / 旧 glibc 宿主
（如 Kylin V10）。rootfs、启动参数、优化补丁、watchdog、压测与排障脚本全部放在本仓库。

> Native, reproducible deployment of **DeepSeek-V4.1-Flash-1017-w8a8** on one
> 8-card Ascend A3 node with vLLM Ascend v0.30.0rc2 — no Docker, just an
> extracted rootfs plus a hardened launcher.

---

## 特性

- **原生 rootfs**：把 `quay.nju.edu.cn/ascend/vllm-ascend:v0.30.0rc2-a3`
  (linux/arm64) 按 OCI 层直接拉取解包，容器镜像里的预编译算子、CANN 9.2.0、
  Python 3.12.13 原样复用，host NPU driver 直连；与机器上其它 vLLM 环境完全隔离。
- **一键可复现**：manifest 解析、whiteout 处理、SHA-256 校验、执行位修复全部脚本化。
- **离线 / 旧 glibc：docker 运行时**（`DSV41_RUNTIME=docker`）：直接使用本机已
  `docker load` 的镜像，vLLM 跑在长驻容器里（宿主 glibc 2.28 也没问题）；无需访问
  registry、无需在宿主安装 Python 3.12。见
  [`docs/offline-deployment.md`](docs/offline-deployment.md)。
- **只清理自己的进程**：服务进程带有 `DSV41_PROC_TAG`，watchdog 重启、`STOP=1`、
  `kill_vllm.sh` 都只处理本部署，不会误杀同机其它 vLLM / NPU 任务。
- **生产级启动参数**：DP2/TP8、Model Runner V2、DSpark K=5、FULL_DECODE_ONLY
  37 个 ACL graph 桶、1M 上下文、prefix caching、`prompt_tokens_details`、
  Engram CPU offload。
- **4 个已实测补丁 + 6 个优化 gate**：默认值固化进源码树，规避 spawn worker
  不继承环境变量导致的静默失效。
- **前缀/ KV 感知路由**：external LB 模式下 DP0/DP1 各自监听独立端口，router 消费
  vLLM KV 事件，按最长 token 前缀把相同 system prompt / 长文档粘到同一 rank；
  默认的 `cost` 策略以 token 为单位给各 rank 打分（未命中 prompt + prefill 排队量 +
  KV 饱和罚分），不再只看请求数，`DSV41_ROUTER_POLICY=legacy` 可回退。详见
  [`docs/router.md`](docs/router.md)。
- **外部 KV 池（Mooncake / AscendStoreConnector，本机暂由 local.env 关闭）**：prefix cache 的
  KV block 扩展进宿主内存（默认 32GB/卡，16 die ≈ 512 GiB ≈ 显存 KV 的 4.6 倍），
  长文档 / 多轮对话 / 共享 system prompt 跨请求、跨 DP rank 命中，避免重复
  prefill；`DSV41_KV_POOL=0` 可一键回滚。**注意**：A3 910C 当前驱动的跨卡
  HIXL 不可用（`halDeviceDisableP2P ret:17`），本机已在
  `config/dsv41.local.env` 关闭该特性；驱动修好后改回 1 即可。详见
  [`docs/kv-pool.md`](docs/kv-pool.md) 第 3.1 节。
- **日志分析**：`scripts/analyze_serve_log.py` 从 serve 日志量化输出 tok/s 的停顿占比、
  两个 rank 的 lockstep 耦合、KV 饱和，用于前后对比（`docs/performance.md` 第 7 节）。
- **无人值守**：`watchdog.sh` 轮询 `/health`，宽限期 + 自动重启 + 日志轮转。
- **可观测 / 可验证**：冒烟测试、2K 吞吐压测、8K 推测解码接受率压测。

## 实测吞吐

单机 8×910C（DP2/TP8），`completions` 接口，约 2K prompt / 256 output，
temperature=0，DSpark 推测解码；**16384 批次 + external LB/router**，
2026-11-02 实测（机器无其它在线流量）：

| 并发 | 单前缀总吞吐 (tok/s) | 双前缀总吞吐 (tok/s) |
|---:|---:|---:|
| 1 | 87.9 | — |
| 2 | 163.7 | 192.1 |
| 4 | 317.5 | 360.4 |
| 8 | 473.6 | 672.9 |
| 16 | 1038.8 | 1084.7 |
| 32 | 1831.6 | **2253.4** |

单前缀是同一热 prompt；双前缀用 `scripts/bench_dual.py` 让两个长前缀分别落到
两个 DP rank，更接近 DP2 实际并发能力。8K→1024 双前缀 @c16 实测约 1.34k tok/s；
完整表和方法见 [`docs/performance.md`](docs/performance.md)。

**2026-11-04 并发压力复测**（未重启服务，与 2026-11-02 相同 2K/256 口径）——
容量与稳定性结论一致：

| 并发 | 口径 | 总吞吐 (tok/s) | 单流 (tok/s) | TTFT (ms) | 失败 |
|---:|---|---:|---:|---:|---:|
| 1 | 单前缀 | 87.7 | 96.5 | 276 | 0 |
| 2 | 单前缀 | 163.1 | 91.0 | 298 | 0 |
| 4 | 单前缀 | 311.7 | 88.3 | 347 | 0 |
| 8 | 单前缀 | 496.6 | 69.6 | 353 | 0 |
| 16 | 单前缀 | 1,044.6 | 73.1 | 347 | 0 |
| 32 | 单前缀 | 1,932.7 | 72.8 | 609 | 0 |
| 16 | 双前缀 | 1,238.4 | 93.6 | 388 | 0 |
| 32 | 双前缀 | 2,123.8 | 82.4 | 665 | 0 |
| 64 | 双前缀 | 3,116.8 | 64.4 | 822 | 0 |
| 64 持续×12 波 | 双前缀 | **3,184.6** | 63.1 | 808 | 0 |

- C=64 已超过单 rank `max_num_seqs=32` 的排队上限，靠 DP2 分摊 + router 亲和消化，
  总吞吐仍随并发增长且持续压测与短时突发持平（无衰减）；
- 压测全程 serve 日志 **0 条失败请求**（`/v1/completions` 全部 200），结束后两 rank
  active 归零、`/router/stats` 无 fallback 激增。

**2026-11-04 批次 A/B（`DSV41_MAX_NUM_BATCHED_TOKENS` 8192 vs 16384）**，
同机同口径（2K prompt / 256 output、cost 路由、引擎前缀缓存热）实测：

| 口径 | 并发 | 16384 (tok/s) | 8192 (tok/s) | Δ |
|---|---:|---:|---:|---:|
| 单前缀（全缓存命中） | 2 | 166.2 | **185.3** | +11.5% |
| 单前缀 | 4 | 323.8 | **346.5** | +7.0% |
| 单前缀 | 8 | 504.6 | **705.4** | **+39.8%** |
| 单前缀 | 16 | 993.8 | **1,055.8** | +6.2% |
| 单前缀 | 32 | 1,809.4 | 1,834.2 | +1.4% |
| 双前缀（冷 prefill 重） | 16 | 1,273.7 | 945.0 | **-25.8%** |
| 双前缀 | 32 | 2,225.8 | 1,679.6 | **-24.5%** |
| 双前缀 C=64 | 突发 | 3,116.8 | 1,730.3 | **-44.5%** |
| 双前缀 C=64 持续×12 波 | 64 | 3,376.9 | 1,794.9 | **-46.8%** |

- 8192 缩小 prefill 分块预算：**冷 prefill 比例高的负载每波需要更多调度步**，
  DP2 lockstep 同步次数翻倍 → 双前缀大跌；decode-only 场景反而吃到停顿变短
  的收益（单前缀 C=8 每流 101 tok/s，8192 比 16384 快 40%）。
- **KV 容量随批次换算**（同 0.90 显存预算）：16384 → 1,268,538 tokens/rank
  （1M 并发 1.21×）；8192 → 2,439,086 tokens/rank（实测，**2.33×**）。
- **生产决策：批次保持 16384**（混合冷 prefill 的真实流量跌幅太大）；
  热前缀为主的业务可显式设 `DSV41_MAX_NUM_BATCHED_TOKENS=8192`。
- 改批次进入编译缓存 key，冷启动 ≈5–9 min。
- Router 侧结论（当日复盘）：`cost` 与 legacy 在引擎正常态下双前缀 C=64
  3,377 vs 3,165，两者相当；当晚的"重启后腰斩"是 **cost 策略缺 max-skew
  护栏** 的 bug（见下），修复后保持 `cost` 为默认，
  回退 `DSV41_ROUTER_POLICY=legacy`。

> **2026-11-04 深夜复盘：当晚"引擎 epoch 异常"根因已定位并修复**。
> 貌似"同配置新引擎吞吐腰斩"实为两个独立问题叠加：
> 1. **router cost 策略缺 max-skew 均衡护栏**（主因，~1.9×）：引擎冷启动后的
>    新 prefill 触发 KV 事件迅速填热 digest 表；digest 温热后 cost 策略
>    把双前缀流量按亲和盲贴 rank0（实测 active 44 vs 20，超出 rank
>    `max_num_seqs=32` 排队）→ 吞吐腰斩至 ~1.05–1.78K。直连 rank 压测
>    （绕过 router，32+32）同引擎实测合计 **3,430**，证明引擎本身无损。
>    修复：`_choose_cost` 加入与 legacy 相同的 `--max-skew` 护栏
>    （active 差超界即溢出均衡），修复后温 digest 态 12 波 C=64 稳定
>    **3,439–3,465**（当晚历史最佳 3,499 系空 digest 首轮）。
> 2. **torch/triton 编译缓存被 8192 实验污染**（次因，~1.6×）：8192 冷启动
>    写入的共享 inductor/triton 摘要被后续 16384 引擎半兼容复用（该引擎
>    启动零缓存写入），典型症状 decode 正常、混合 prefill 慢。清空
>    `~/.cache/vllm/torch_compile_cache` 与 `~/.triton/cache` 后冷启动
>    重编，router 路径由 ~1.05K 恢复至 ~1.77K（乘上护栏修复回到 3.4K+）。
>    建议：改批次做 A/B 前先备份这两个缓存目录。

冷启动（权重加载 + KV cache + 图捕获）约 **5–9 分钟**；权重每 die 约
38.7 GiB。16384 批次下激活峰值约 8.2 GiB，KV cache 实测 6.7 GiB
（**1,268,538 tokens / DP rank**，1M 上下文并发 1.21×）；8192 批次实测
**2,439,086 tokens / DP rank（2.33×）**。32768 在 1M 上下文下放不进 55 GiB 预算。测量方法与
取舍见 [`docs/performance.md`](docs/performance.md)。

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
.
├── config/
│   ├── dsv41.env.example        # 所有可调参数的示例配置
│   └── dsv41.docker.env.example # docker 运行时 / 离线节点预设
├── docs/
│   ├── offline-deployment.md    # 离线 + 旧 glibc 宿主（docker 运行时）完整部署指南
│   ├── deployment.md            # 从零部署（rootfs、权重、补丁、启动、验证）
│   ├── architecture.md          # 进程/端口/rootfs 隔离与请求链路
│   ├── router.md                # external LB / KV 前缀感知路由
│   ├── performance.md           # 实测数据、参数取舍、压测方法
│   ├── patches.md               # 补丁与优化 gate 详解
│   ├── troubleshooting.md       # 故障速查表
│   └── api.md                   # OpenAI 兼容 API / SDK / Claude Code 接入
├── examples/
│   ├── curl_chat.sh
│   └── openai_chat.py
├── scripts/
│   ├── common.sh                # 共享配置与进程工具
│   ├── container.sh             # docker 运行时：创建/管理长驻容器（up/exec/shell/down…）
│   ├── setup_rootfs.sh          # 拉取 OCI 镜像 → rootfs（处理 whiteout）
│   ├── oci_extract.py           # 上述脚本的提取实现（仅标准库）
│   ├── apply_patches.sh         # 应用补丁 + 固化 gate 默认值
│   ├── doctor.sh                # 只读预检（硬件/rootfs/权重/补丁/端口）
│   ├── serve.sh                 # 启停 / 状态
│   ├── watchdog.sh              # 健康检查 + 自动重启
│   ├── kill_vllm.sh             # 彻底清理 vLLM / NPU worker
│   ├── patch_utils.py           # 补丁状态判定（含 gate 默认值归一化）
│   ├── prefix_router.py         # external LB 前缀 / KV 感知路由器
│   ├── analyze_serve_log.py     # serve 日志分析（停顿 / lockstep / KV），只读
│   ├── bench_2k.py              # 2K prompt 吞吐（单前缀）
│   ├── bench_8k.py              # 8K prompt + 接受率（单前缀）
│   ├── bench_dual.py            # 双前缀 DP2 并发吞吐
│   └── patches/                 # 补丁权威副本
├── systemd/                     # 可选 systemd 单元（含 dsv41-container.service）
├── tests/
│   ├── smoke_test.py            # 在线冒烟测试
│   ├── test_container.py        # 容器命令组装 / 进程清理范围单元测试
│   ├── test_analyze_serve_log.py # 日志分析脚本单元测试
│   ├── test_router_logic.py     # router 前缀/路由/打分/引用计数单元测试
│   ├── test_router_proxy.py     # router 代理路径测试（假 vLLM rank，无需 NPU）
│   └── test_tools.py            # 离线工具单元测试
└── requirements.txt             # 客户端/压测可选依赖
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

# 4) 应用 4 个优化补丁（幂等）
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
| [kv-pool.md](docs/kv-pool.md) | 外部 KV 池（Mooncake / AscendStoreConnector）：容量、内存预算、启用与回滚 |
| [patches.md](docs/patches.md) | 4 个补丁、6 个 gate、为什么默认值必须是 1 |
| [troubleshooting.md](docs/troubleshooting.md) | 症状 → 原因 → 修复 |
| [api.md](docs/api.md) | OpenAI 兼容接口、SDK、Claude Code |

## 许可证与致谢

本仓库以 **Apache-2.0** 发布。补丁修改的文件来自
[vLLM](https://github.com/vllm-project/vllm) 与
[vLLM Ascend](https://github.com/vllm-project/vllm-ascend)（Apache-2.0）。
模型权重由 Eco-Tech 发布（模型卡片标注 MIT License），本仓库不分发权重。
完整声明见 [`NOTICE`](NOTICE)。
