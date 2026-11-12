# 不改批次参数的工程杠杆矩阵（lever scout / task-6）

> 作者：lever-scout（task-6）｜时间：2026-11-12 14:5x｜**未启动/未重启任何引擎，未做压测**
> 权威源码树（远端只读）：`/cache/dsv41-rc2-rootfs/vllm-workspace/{vllm-ascend,vllm}`
> 本地镜像 `/tmp/src`：已用 md5 与远端逐文件核对一致（`ascend_config.py` / `compilation.py` /
> `dflash/speculator.py` / `speculative.py` 四个文件 md5 完全相同），下文 `src:` 行号两边通用。
> 参考仓文档快照：`/tmp/refs/a/docs`（repo1，228 篇）、`/tmp/refs/b`（repo2）。

---

## 0. 一句话结论

**在 BAT=16384 / max_num_seqs=32 / 图桶≤192 三项锁死的前提下，配置层已经没有量级 ≥5% 的杠杆了。**
本矩阵里最值得先跑的 5 条，期望值分别是：

| 排名 | 杠杆 | 期望收益（诚实区间） | 成本 | 置信度 |
|---|---|---|---|---|
| 1 | draft 图缺桶补齐（+7 张 draft 图，目标桶不变） | 对应 batch 档 decode 聚合 **+1~3%**；单流 **0** | 起服 +~4s | 【推断·高】（数学+实测捕获数双证） |
| 2 | `multistream_engram_overlap=true` | 每步 **−0.05~0.2 ms（0.1~0.5%）** | env-only | 【推断·中】 |
| 3 | `mc2_comm_alg=fullmesh_v2`（A3 能力项） | 每步 **−0~3%**（可能 0 或负） | env-only | 【推断·低】 |
| 4 | HCCL_BUFFSIZE 实测（★ 现状从未生效） | ms/step **≈0**；可能回收 HBM | env-only | 【推断·高（负）】 |
| 5 | limit_core 预算细扫（main20/aux28、aux16） | **≤±1%**（噪声级；aux16 单流曾有 106.8 tok/s 记录） | env-only | 【实测·文档】 |

另外发现 **3 个配置事实性问题**（不是性能杠杆，但会误导后续实验，见 §2）：
① `HCCL_BUFFSIZE` 在生产里**从未被设置**；② `TASK_QUEUE_ENABLE=1` / `OMP_NUM_THREADS=1`
其实已在节点 pid1 环境里（并非「我们没设」）；③ `limit_core_num` 的真实收益是 ≈+1%，
而 `scripts/serve.sh` 注释写的是 "+3~6%"。

---

## 1. 口径、基线与约束

- 生产锁死参数：`BAT=16384`、`max_num_seqs=32`、`cudagraph_capture_sizes` ≤192。
  **本矩阵所有 env overlay 一律不动这三项**（唯一改动桶表的是 exp51——只在列表里插入 7 个 raw 值，
  目标桶集合数学上逐个验证不变，见 §3.1）。
- 基线（`/cache/exp/summary.csv` + `docs/performance.md`，均为【实测】）：
  单流 100–106 tok/s、40.0–41.2 ms/step、TTFT 249–310 ms、接受率≈0.55–0.58、
  双前缀 c16≈910–1010 tok/s、c64≈1683–1917 tok/s；`num_preemptions_total=0`（exp0c/0d/0e metrics）。
- 起服内存事实（`logs/serve_dsv41w8a8rc2-20261112_132648.log`，【实测】）：
  61.27 GiB/die、weights 38.74 / 激活峰值 8.18 / 非 torch 1.49 / 图 0.68 GiB，
  **KV 只有 6.72 GiB**；worker 自己建议「吃满空闲可到 `--kv-cache-memory=11.47 GiB`」。
  目标图捕获 **32/32**、draft 图 **25/25**；cpu_bind `mode=global_slice`；
  engram 表 host 内存 **12.87–12.88 GiB/rank**。
- 评测口径纪律（沿用 `docs/performance.md`）：跨进程只能比 **ms/step / 接受率**；
  聚合吞吐受接受率抽签影响，必须固定 prompt/temperature/并发；噪声级改动（<1.5%）
  要 ≥5 rep 中位，3 rep 不够。

### 1.1 怎么用这些 overlay

```bash
# 测试节点（run_config.sh 会把 overlay 用 `set -a` source，再按 DSV41_PROC_TAG 清理进程）
bash scripts/experiments/run_config.sh exp51_draft_buckets scripts/experiments/envs/exp51_draft_buckets.env
```

三条必须知道的机制（否则会误判「改了没生效」）：

1. `run_config.sh:56-59` 先 export 生产基线 env，**再用 `set -a` source overlay** ⇒ overlay 覆盖优先。
2. 测试副本 `/cache/exp/repo-test/scripts/serve.sh` 被 `scripts/experiments/patch_serve.py` 打了 7 个
   knob（`DSV41_CUDAGRAPH_MODE / STATIC_KERNEL / NPUGRAPH_EX / PREFILL_MC2 / FUSED_MC2 /
   MULTISTREAM_SHARED_EXPERT / DRAFT_GRAPH`）；生产 `scripts/serve.sh` 没有这些变量——
   所以 **overlay 只在实验链路可用，不代表生产可注入**。
3. `serve.sh:373-378` 的 `DSV41_EXTRA_SERVE_ARGS` 是**按空格裸切**的：写 JSON 时不能有空格；
   且重复 `--additional-config` / `--speculative-config` 时 argparse「后者覆盖」（`store` 动作），
   所以这类 overlay 必须在 EXTRA 里给出**完整 JSON**（本目录已按基线键写全）。

---

## 2. 先纠正 3 个配置事实（会影响后续实验设计）

### 2.1 ★ `HCCL_BUFFSIZE` 在生产从未生效（配置 bug）【实测·env】

- `repo:scripts/serve.sh:45` 声明 `DSV41_HCCL_BUFFSIZE:=1024`，
  但 `repo:scripts/serve.sh:238` 写的是**裸 `export HCCL_BUFFSIZE`**（没有 `HCCL_BUFFSIZE=$DSV41_HCCL_BUFFSIZE`
  的赋值）。bash 语义：变量未设置时 `export X` 只是把「未设置的空变量」标记为导出 —
  实测 `env -u HCCL_BUFFSIZE bash -c 'export HCCL_BUFFSIZE; env|grep -c ^HCCL_BUFFSIZE='` → `0`。
- 远端证据：`/proc/1806051/environ`（当前测试实例引擎进程）里只有
  `HCCL_OP_EXPANSION_MODE=AIV`、`TASK_QUEUE_ENABLE=1`、`OMP_NUM_THREADS=1`、`VLLM_USE_V2_MODEL_RUNNER=1`、
  `LD_PRELOAD=…jemalloc…`，**没有 `HCCL_BUFFSIZE`** ⇒ 实际跑 HCCL 默认值，
  而 `config/dsv41.env.example:98`、`docs/deployment.md:197` 都写着 1024。
- 影响：所有「HCCL_BUFFSIZE 已设 1024」的假设都不成立；但它也意味着
  **512/1024/2048 三条 A/B 是干净的**（overlay 里直接 `HCCL_BUFFSIZE=512` 即可，
  overlay 被 `set -a` source 后 serve.sh 的裸 export 会把这个已存在的值原样导出）。
- 建议（不在本人写入范围，交 Lead 决策）：把 `serve.sh:238` 改成
  `export HCCL_BUFFSIZE="$DSV41_HCCL_BUFFSIZE"`，或干脆删掉这个假开关并在文档里写「不设置」。

### 2.2 `TASK_QUEUE_ENABLE=1` / `OMP_NUM_THREADS=1` 其实已经开着【实测·env】

- 远端节点 **pid1（systemd）** 的环境里就有 `TASK_QUEUE_ENABLE=1` 与 `OMP_NUM_THREADS=1`
  （`/proc/1/environ`），所有引擎进程继承；当前实例 `/proc/<engine>/environ` 可见。
- ⇒ 「repo2 设了 TASK_QUEUE_ENABLE=1，我们没设」这个前提不成立：运行时是同值。
  exp60 只是把它写死做保险（防止将来从干净 systemd 环境起服丢失）。
- `OMP_NUM_THREADS>1` 不要设：vLLM 启动时用 `available_cpu_count() // local_world_size`
  给 worker 起线程（`src:vllm/v1/executor/multiproc_executor.py:1100-1125`），
  进入稳态后**主动把 torch 线程降到 1**，外部设 >1 会触发 spin-wait 争用告警
  （`src:vllm/utils/torch_utils.py:259-295`）。repo2 的 `OMP_NUM_THREADS=10` 在本栈是负向或无效。

### 2.3 `limit_core_num` 的真实收益是 ≈+1%，serve.sh 注释里的 +3~6% 偏乐观【实测·文档】

- `docs/optimization-2026-11-11.md:264-278` 明确写「实测 ≈+1%，在噪声内（诚实负结果）」；
  `docs/optimization-deep-dive-2026-11-12.md:232-233` 给出三组数据：
  main16/aux32 = 105.1–105.4 tok/s、aux16 = 106.8 tok/s（但 TTFT 310 ms、双前缀 c16 1716）。
- `repo:scripts/serve.sh:283-291` 的注释写 "+3~6%"——以后引用请以 docs 的 ±1% 为准。

---

## 3. 杠杆详情（按建议执行顺序）

排序原则：**先零风险确认项（可能直接可交付）→ 再可能为 null 的通信/主机项 → 最后探索项**。
每条都给了现成 overlay（`scripts/experiments/envs/exp5x_*.env`，均为新增文件，未覆盖任何既有文件）。

### 3.1 【P0-1】draft 图缺桶补齐 → `exp51_draft_buckets.env`

**机理（已闭环，非猜测）**
- 目标模型 decode 每步 token 数 = `num_reqs × (1+K) = 6N`；`FULL_DECODE_ONLY` 下每个 raw 桶
  向上取整到 `decode_query_len`（`src:vllm/v1/worker/gpu/cudagraph_utils.py:255,291-336`）。
- 现有 37 个 raw 桶对目标取整后去重 = **32 个 {6,12,…,192}**（实测日志 `32/32`），
  对 draft（`decode_query_len = num_query_per_req = 5`，`src:.../spec_decode/dflash/speculator.py:56,141-146`；
  `dspark/speculator.py:50-56` 在 `sample_from_anchor=true` 时取 `num_speculative_steps`）取整后
  **只剩 25 个**（实测日志 `25/25`）。
- 我按源码重算：raw≥168 的 draft 桶因 `num_reqs > max_num_reqs(32)` 被丢弃 ⇒ draft 桶只覆盖 ≤160；
  5 的倍数在 5..160 里缺 **45,70,85,95,110,125,135**，恰好与实测 25 vs 32 的差一致。
- 补这 7 个 raw 值后：目标取整 `round_up(45..135,6) = 48,72,90,96,114,126,138`，
  **全部已在目标 32 桶集合里**（脚本逐项验证，见下表）⇒ **目标图不多捕一张、桶上限仍 192**。

| raw 新增 | 45 | 70 | 85 | 95 | 110 | 125 | 135 |
|---|---|---|---|---|---|---|---|
| 目标桶 round_up(,6) | 48 | 72 | 90 | 96 | 114 | 126 | 138 |
| 目标桶是否已存在 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| draft 桶 round_up(,5) | 45 | 70 | 85 | 95 | 110 | 125 | 135 |
| 受影响 num_reqs | 9 | 14 | 17 | 19 | 22 | 25 | 27 |

**预期**：单流（num_reqs=1）0；num_reqs∈{9,14,17,19,22,25,27} 的 decode 步不再把 draft pad 到上一档
（最差 −11% draft token）⇒ 这些档位的聚合吞吐 +1~3%。双前缀 c16/c64 进出批次的爬升/回落阶段会经过这些档位。
**风险**：低（只多 7 张 draft 图，捕获 +~4s、显存 +几十 MB）。
**验证**：起服日志 `Capturing dspark CUDA graphs (FULL): … 32/32`（基线 25/25），目标仍 32/32；
bench 单流 + 双前缀 c16/c64。
**依据**：`src:vllm/v1/worker/gpu/cudagraph_utils.py:255,314`；`src:.../spec_decode/dflash/speculator.py:56,141-146`；
`src:.../spec_decode/dspark/speculator.py:50-56`；`src:vllm_ascend/worker/v2/aclgraph_utils.py:108-121`；
`logs/serve_dsv41w8a8rc2-20261112_132648.log`。

### 3.2 【P0-2】engram 查表换辅助流 → `exp52_engram_overlap.env`

**机理**
- Engram 表走 `cpu_offload=true`：host 内存 `aclrtMallocHost` + `aclrtHostRegisterV2(PINNED|MAPPED)` +
  `aclrtHostGetDevicePointer`，NPU gather kernel 直接读 host（无 H2D copy）
  （`src:vllm_ascend/models/deepseek_v41/engram/npu.py:1-9,135-190`）。
- A3/TP8 下 `can_capture_engram_producer()` 要求 `table.dp_size==1 and table.tp_size==1`，
  而 `embed_tokens.tp_size = TP=8`（`engram/embedding.py:99`）⇒ **FULL 图内没有 producer**，
  每步 hash + gather 在 Python 侧主 stream 同步做完再 replay（`model.py:1259-1273`）。
- `multistream_engram_overlap` 默认只在 `device_adaptor_family == FP8_OPTIMIZED`（A5）打开；
  A3 的 profile 是 `STANDARD` ⇒ **当前为 False**（`src:vllm_ascend/ascend_config.py:573-576`,
  `device/hardware_profile.py:301-318`）。打开后走 `_engram_prepare_stream` +
  按桶缓存 `ExternalEvent`，主图 consumer 等待事件（`model.py:1073,1247-1257,1433-1445`,
  `engram/model_state.py:130-150`）。

**预期**：每步把（hash kernel + DP gather + 2 层 host-UVA lookup）藏到 metadata 准备后面，
【推断】−0.05~0.2 ms/step（0.1~0.5%），TTFT 微降；高并发按步线性。
**风险**：中。ExternalEvent 与 FULL 桶绑定，跨流同步一旦错会静默错值 ⇒ 必须同时回归
（文本、接受率 A 值、tok/s）；`dp_shared_memory=true` 保持不变。
**验证**：单流 ms/step ≥3 rep + 双前缀 c16/c64；日志无 event/lookup 报错；
接受率与基线一致（0.567±0.01）。
**只读观测方案（不起第二个引擎）**：先用 exp52 的 on/off 差值作为「engram host 表每步开销」的
上界；若差值为 0，则本项判负、且可断定 host-UVA 表不是单流瓶颈。
补充只读事实：起服日志 `Engram table offloaded … 12.87 GiB per rank`（`embedding.py:124`）；
worker 因 cpu_offload 会**跳过进程级 NUMA 页面迁移**（`src:vllm_ascend/worker/worker.py:944-953`
的 `migrate_memory=not cpu_offload`）——所以「进一步 CPU/NUMA 收紧」这条天然被关掉，见 §5。
**依据**：同上 + `extra-info/code-opportunities.md:§2.2`（图外段清单，估算 0.1–0.5 ms）。

### 3.3 【P0-3】KV 容量微调（不动桶）→ `exp55_kv_util_093.env`、`exp56_kv_util_095.env`

**机理**：图桶是**显式列表**（`repo:scripts/serve.sh:324`），`gpu_memory_utilization` 只决定 KV blocks，
既不进桶表也不改编译 key ⇒ 调它**不会触发任何桶改动**。
起服日志（`worker.py:933`）已给出量化：当前 KV 6.72 GiB，"吃满空闲" 可到 **11.47 GiB**；
0.93 约 +1.8 GiB（token 容量 +~26%），0.95 约 +3 GiB（+~45%）。
**预期**：单流/decode ms/step **0**；只在长上下文/高并发容量与 preemption 上起作用。
诚实提醒：现有 bench 口径 `num_preemptions_total=0`（exp0c/0d/0e metrics）⇒ **本项的 bench 结论大概率是 null**；
它的真实价值是给 1M 上下文生产流量留存余量，不是提 tok/s。
**风险**：中（HBM 只剩 worker 自留的 150 MiB 安全垫；任何新图/新算子可能 OOM）。一次只 +0.03。
**验证**：起服日志 `Available KV cache memory` / `Free memory on device`；
压测期间 `vllm:num_preemptions_total`、`vllm:kv_cache_usage_perc`；OOM 立刻回 0.90。
更精确的替代（同样不改桶）：`--kv-cache-memory-bytes`（跳过 memory profiling，
`src:vllm/vllm/config/cache.py:229-236`、`src:vllm_ascend/worker/worker.py:572-578`），
但要自己承担 activation 峰值估算风险。
**依据**：`logs/serve_…132648.log`（worker.py:933）；`src:vllm_ascend/worker/worker.py:897-924`。

### 3.4 【P1-1】HCCL_BUFFSIZE 三档 → `exp57/58/59_hccl_buffs_*.env`

**机理**：HCCL 收发缓冲（MB）。repo1 的受控扫描【实测】表明 61 KB→2 MB 耗时几乎不变
（49.7→84.3 µs，落在**固定延迟区**），而我们的 decode 消息是 61 KB（N=1）~491 KB（N=8）
⇒ 调缓冲大小**不改单次时延**（`/tmp/refs/a/docs/HCCL-ALLREDUCE-NEGATIVE-20261005.md:§1`）。
**为什么仍建议跑**：(a) 现状是 HCCL 默认值（§2.1 bug），三档 A/B 便宜且能一次性把这条线钉死；
(b) 更小缓冲可能回收 HBM（KV 6.72 GiB 很紧），间接有益。
**预期**：ms/step ≈0；512 若让 `Available KV cache` 变大则算附带收益；2048 大概率吃 HBM 判负。
**风险**：低~中（HBM；prefill 大包是否分块变多会影响 TTFT，需一起看）。
**依据**：同上 + `/tmp/refs/a/docs/A3-DECODE-BANDWIDTH-AND-CACHE-20261006.md:127` +
`repo:scripts/serve.sh:45,238`。

### 3.5 【P1-2】主机侧确认项 → `exp60_taskq_lockin.env`、`exp61_malloc_conf_jemalloc.env`

- **TASK_QUEUE_ENABLE**：已实测节点级 =1（§2.2）⇒ 预期 0；exp60 只为固化。
  兼容性【实测】：`=1 CAPTURE_OK/REPLAY_OK 288.5 µs/replay`，`=2` 直接 CAPTURE_FAIL
  （`/tmp/refs/a/docs/V41-DCP8-DECODE-PERF-20261002.md:283-296`）。
- **jemalloc `MALLOC_CONF` 微调**：jemalloc 已 LD_PRELOAD（`repo:scripts/serve.sh:248-253`，
  docs 实测 −3.2~5.1%/step）；再调 dirty/muzzy decay 与背景线程属【推断】0~1%，
  噪声级改动要 ≥5 rep。CPU 预算已按 rank 切开（`global_slice`），背景线程别抢主线程。
- **CPU 绑核/NUMA 进一步收紧**：**没有 env 杠杆**。A3 的 `cpu_binding_mode` 由硬件 profile 写死为
  `GLOBAL_SLICE`（`src:vllm_ascend/device/hardware_profile.py:301`），`cpu_binding.py` 里没有任何
  env override（只有 `ASCEND_RT_VISIBLE_DEVICES` 读取）；且 engram cpu_offload 时
  `worker.py:951` 明确 `migrate_memory=False`（不迁移进程页，避免碰 pinned host 表）。
  ⇒ 见 §5「已排除」。
- **hugepages / vm.swappiness**：远端实测 `HugePages_Total=0 / nr_hugepages=0`，
  `AnonHugePages=41 GB`（THP 已在用），`vm.swappiness=0` 已经是 0
  ⇒ 没有可通过 env 获得的收益；显式 hugetlbfs 需要 root + 预留巨页，且 engram/KV 池都是
  pinned 内存，收益路径不清晰。⇒ 排除（§5）。

### 3.6 【P1-3】spec decode 概率草案 → `exp62_draft_probabilistic.env`

**机理**：默认 `draft_sample_method="greedy"`（draft 概率按 one-hot 近似，`src:vllm/vllm/config/speculative.py:588-593`）；
`probabilistic` 用完整 draft 分布做 gumbel 采样 + 概率比检验（DSpark 路径已实现，
`src:.../dspark/speculator.py:126-155`），extra 缓冲 ≈32×5×129280×2B ≈ **41 MB/die**。
**预期**：【推断】temperature>0 流量接受率 +1~4%；temperature=0 流量无语义变化、只白付采样开销
（draft 段 ms/step 可能 +0.2~1 ms）。净收益必须实测。
**验证口径（关键，别用错脚本）**：`scripts/bench_2k.py:86`、`bench_dual.py:82`、
`scripts/experiments/bench_suite.py:115` **全部硬编码 temperature=0** ⇒ 不能评估本臂。
需要一段 temp=0.8~1.0 的小客户端（固定 prompt、固定 max_tokens），用
`Δvllm:spec_decode_num_accepted_tokens_total / Δvllm:spec_decode_num_draft_tokens_total`
与 `Δvllm:spec_decode_num_drafts_total` 推 tokens/step，再看 ms/step；同一进程内先后跑
temp=0 与 temp>0 两段，**temp=0 段必须与基线一致**（回归护栏）。
**风险**：中（显存 +41 MB/die；与 `use_local_argmax_reduction` 互斥——我们没开；
与 heterogeneous vocab 限制无关）。
**依据**：`src:vllm/vllm/config/speculative.py:588-593`；`src:.../dspark/speculator.py:126-155`；
`src:vllm/vllm/config/vllm.py:1154-1157`（watermark 前提，不触发）。

### 3.7 【P2-1】MoE decode 通信算法 → `exp64_mc2_fullmesh_v2.env`

**机理**：decode 每层 2 次 EP16 dispatch/combine（40 层 ⇒ 80 次/步，
`src:vllm_ascend/ops/fused_moe/token_dispatcher.py:202-274`），`mc2_comm_alg` 透传给
`npu_moe_distribute_*`；A3 能力集含 `MC2_FULLMESH_V2_COMM`
（`src:vllm_ascend/device/hardware_profile.py:~275-310`）、校验只要求该能力
（`ascend_config.py:981-989`）。默认 `""` 走 op 默认算法。
**预期**：【推断，未验证】ms/step −0~3%（也可能 0/负）。这是本清单里少数可能超过 1% 的方向，
但完全没在这个栈上验证过。
**风险**：高（通信算法更换 ⇒ 数值回归 + A 值 + 文本；建议与 MegaMoe 分开跑）。
`mc2_comm_alg=hierarchy` 在 A3 不可用（无 `MC2_HIERARCHY_COMM`）⇒ 不要试。
**依据**：`ascend_config.py:579-605,981-1006,1285-1295`；`token_dispatcher.py:202-274,390`。

### 3.8 【P2-2】DP padding 同步改 device → `exp63_dp_device_ar.env`

**机理**：async scheduling 下 vLLM **强制** DP padding 同步走 CPU Gloo
（`src:vllm/vllm/config/vllm.py:1489-1500`），每步 1 次 4×2 int32 allreduce 在调度线程阻塞
（`src:vllm/vllm/v1/worker/dp_utils.py:36-78`）。显式 `--no-disable-nccl-for-dp-synchronization`
可改回 device 组。
**预期**：【推断】±0.05 ms/step；两 rank 负载接近时可能略快，但 async 的设计前提就是避开
device 同步点（会破坏 draft-propose 与 output 的重叠）⇒ 可能更慢/更抖。
**风险**：中高（异步流水被打断；不要与其它变量混跑）。
**依据**：`vllm/config/parallel.py:238-242`；`vllm/engine/arg_utils.py:1230-1233,355-363`
（BooleanOptionalAction ⇒ `--no-…` 形式有效）。

### 3.9 【P2-3】调度侧 prefill 预算 → `exp66_long_prefill_4k.env`

**机理**：`--long-prefill-token-threshold`（`src:vllm/vllm/config/scheduler.py:70-72`，
CLI `arg_utils.py:1608-1610`）= 长 prompt 单 chunk 上限，**不是批次参数**（BAT 仍 16384）。
BAT=16384 时一个长 prefill 步会占满整步预算，把同批 decode 卡在后面。
**预期**：【推断】高并发 TTFT p95 / decode 步抖动改善；单流 TTFT 变差（多切几步）。
短 prompt 流量下几乎无影响。
**风险**：低（调度器开关，不改图/桶）；但 TTFT 口径会变，评测必须标注。
**依据**：同上。

### 3.10 【P2-4】KV 池对 decode 的复核 → `exp65_kvpool_off_recheck.env`

**源码已定案**【实测·源码】：backend=memcache 且未开 `save_decode_cache`/`layerwise_offload` 时，
`pool_scheduler._process_running_cached_request` 对 decode 请求直接 `return None`
（`src:vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:955-961`）
⇒ **decode 阶段不产生 save/load**，开关 KV 池不应影响 ms/step；差异只可能在 TTFT。
**复核口径（只读，先做这个，再决定要不要起 A/B）**：
1. 在「纯 decode 窗口」（长输出、不同时发新 prompt）前后抓 `/metrics`，
   看 `vllm:ascend_store_load_get_duration_seconds` 的 `_count/_sum` 是否增长 → 应为 0；
2. 同一进程内对比「本地前缀全命中」vs「冷前缀」请求的 ms/step，应一致；
3. 真要 A/B 再用 exp65 跑基线口径，只比 ms/step 与接受率（**不要拿 TTFT 说事**——
   关池会去掉外部前缀命中，TTFT 一定变差）。
**依据**：`pool_scheduler.py:955-961`；`ascend_store/metrics.py:58-120`（指标名）；
`repo:scripts/serve.sh:283-296`；`extra-info/code-opportunities.md:§2.2`。

---

## 4. 建议的执行顺序（Lead 统一串行）

| 波次 | 实验 | 判据（继续/放弃） | 预计单次耗时 |
|---|---|---|---|
| 1 | exp51 draft 桶 | 日志 32/32 draft；双前缀 c16/c64 不退化即收录 | 起服 + 捕获 ~4s + bench |
| 1 | exp60 taskq lock-in | ms/step 与 exp40 基线一致（应一致）⇒ 固化文档 | 一次起服 |
| 2 | exp52 engram overlap | 单流 ms/step 改善 ≥0.1 ms 且接受率不变 | 一次起服 |
| 2 | exp55 kv_util 0.93 | 无 OOM、KV↑、ms/step 不变 ⇒ 作为容量保险可留 | 一次起服 |
| 3 | exp57/58 HCCL 512/1024 | 只有 KV↑ 或 ms/step 改善才继续 2048 | 两/三次起服 |
| 3 | exp53/54 limit_core | 与 exp40 差 <1% 则维持现状（16/32） | 两次起服 |
| 4 | exp64 mc2 fullmesh_v2 | 先跑一次判方向（含数值回归） | 一次起服 |
| 4 | exp63 DP device AR | 只在与基线同口径的单流上比 | 一次起服 |
| 5 | exp62 probabilistic draft | **需要 temp>0 客户端**，否则不要跑 | 一次起服 + 自写负载 |
| 5 | exp66 long prefill / exp65 KV 池复核 | 按生产流量形态决定 | 低频 |

> 起服之间务必遵守 `docs/optimization-2026-11-11.md:§7.5`：**快速连续重启 10–15 次会把节点
> HCCL 打进异常态**（EI0015），每轮之间留足间隔；`run_config.sh` 已有等待 NPU 释放的逻辑。

---

## 5. 已排除 / 不要浪费时间（含原因与依据）

| 项 | 结论 | 原因 / 依据 |
|---|---|---|
| async scheduling | ❌ 已开，不要再加 | 日志 `Disabling NCCL for DP synchronization when using async scheduling`；`extra-info/code-opportunities.md:§4` |
| `HCCL_OP_EXPANSION_MODE=AICPU` | ❌ 起不来 | 8 worker 全 `WorkerProc failed to start`（`/tmp/refs/a/docs/HCCL-ALLREDUCE-NEGATIVE-20261005.md:§2`） |
| `HCCL_ENABLE_OPEN_AIV=1` | ❌ 小消息慢 2× | 同上 §2（61 KB 190.8 vs 89.1 µs） |
| `HCCL_INTRA_PCIE/ROCE_ENABLE=0` | ❌ 无差异（首测假象） | 同上 §2，预热不足的 ±100% 假象教训 |
| `HCCL_ALGO` / `HCCL_DETERMINISTIC=1` | ❌ 不建议 | allreduce 首要矛盾是「调用次数 81 次/步」而非算法；deterministic 通常更慢且无收益路径；repo1 已把通信旋钮判负 |
| `TASK_QUEUE_ENABLE=2` | ❌ 禁止 | 图捕获直接报 `ERR00007 ... Do not support TASK_QUEUE_ENABLE = 2 during NPU graph capture`（repo1 实测） |
| `OMP_NUM_THREADS=10`（repo2 值） | ❌ 负向 | 稳态 torch 线程必须为 1，外部 >1 会 spin-wait 争用（`src:vllm/utils/torch_utils.py:259-295`）；节点已是 1 |
| `enable_npugraph_ex` / `enable_static_kernel` / `enable_super_kernel` | ❌ 两条独立原因 | ① A3 硬件 profile 不含 `NPUGRAPH_EX`，`ascend_config.py:151-166` 会强制降级；② 实测 `ERR00100 PTA call acl api failed` 起不来（`docs/optimization-2026-11-11.md:94`） |
| `cudagraph_mode=FULL_AND_PIECEWISE` | ❌ 已实测负结果 | TTFT 无改善、双前缀 c2/c16 从 197/1333 → 159/898（**−19%/−33%**），`docs/optimization-2026-11-11.md:92` |
| `enable_prefill_mc2=1` @BAT16384 | ❌ OOM | MC2 buffer 按 BAT 分配，1M 上下文 + 16384 下 HBM 不足（`docs/optimization-2026-11-11.md:95`；降 BAT 才能试，但 BAT 锁死） |
| `enable_fused_mc2=1` | ❌ 输出退化 | 单流只返回 1 token、spec 指标消失（`docs/optimization-2026-11-11.md:93`）；MegaMoe(`=2`) 另有专门实验线 |
| `mc2_comm_alg=hierarchy` | ❌ 硬件不支持 | A3 能力集无 `MC2_HIERARCHY_COMM`，`ascend_config.py:990-994` 直接 raise |
| `enable_eplb` / `dynamic_eplb` | ❌ 结构性受阻 | `VllmEplbAdaptor` 假定各 MoE 层同构，而 DSpark draft(128 experts) 与主模型(384) 注册进同一张表 ⇒ `IndexError`（`/tmp/refs/a/docs/HCCL-ALLREDUCE-NEGATIVE-20261005.md:§3.6`） |
| `enable_force_eplb` | ❌ 纯诊断 | 替换 topk_ids 但权重不变 ⇒ 输出错误（同上 §3.6） |
| `use_local_argmax_reduction` | ❌ 缺实现 | 需要 draft 模型 `get_top_tokens()`，`models/deepseek_v41/dspark.py` 没有（`extra-info/code-opportunities.md:§5`） |
| `dynamic_spec_config`（Ascend 自有） | ❌ V2 runner 无消费点 | `worker/v2/**` 无读取点（`extra-info/code-opportunities.md:§5`） |
| prefix caching hash algo（xxhash/xxhash_cbor） | ❌ 依赖缺失 | rootfs 里 `import xxhash → ModuleNotFoundError`（实测）；sha256→sha256_cbor 无性能理由；2K prompt 只有 16 个 block，CPU hash 成本可忽略 |
| 关 KV events（`DSV41_KV_EVENTS=0`） | ❌ 不要 | 事件只在 block add/remove 时 publish（`src:vllm/vllm/v1/core/block_pool.py:275-307,529,609-613`），不在 decode 每步；关掉反而丢 router 前缀亲和 |
| 关 prefix caching | ❌ 不要 | 本地前缀命中率 ≈94%（`extra-info/code-opportunities.md:§6`），router 也依赖 |
| hugepages / vm.swappiness | ❌ 无可获得收益 | 远端实测 `HugePages_Total=0,nr_hugepages=0`、`AnonHugePages=41 GB`（THP 已在用）、`swappiness=0` 已是 0；需要 root+预留巨页，路径不清晰 |
| CPU 绑核/NUMA 进一步收紧 | ❌ 无 env 杠杆 | A3 `cpu_binding_mode=GLOBAL_SLICE` 写死（`hardware_profile.py:301`）；`cpu_binding.py` 无 env override；engram offload 时 `worker.py:951` 明确不做页面迁移 |
| `enable_transpose_kv_cache_by_block` / `enable_mlapo` / `multistream_dsv4_dsa_overlap` / `weight_nz_mode` | ✅ 已是默认开（1/True），无需设置 | `src:vllm_ascend/ascend_config.py:569-618` |
| `VLLM_USE_BREAKABLE_CUDAGRAPH=1` | ❌ 不建议 | Ascend 明确 opt-in 且默认 0（`platform.py:71-77`）；它主要服务 PIECEWISE，而 PW 已实测负结果 |
| KV cache dtype int8 / `enable_kv_nz` | ⚠ 后备（容量杠杆，非本矩阵） | 能把 KV 减半 → 容量翻倍，但数值/attention kernel 风险高，且 `enable_kv_nz` 仅 MLA 且需 `enable_transpose_kv_cache_by_block` 校验（`ascend_config.py:892-906`）；要做应单独立项 |

**邻接（代码级，不属本矩阵 env-only 范围，另有负责人在做）**：
`scripts/patches/experimental/0301-qnorm-wq-b-fuse.patch`、`0302-slotmap-skip-eager.patch`、
`0303-oproj-pad-empty.patch`、`0310-megamoe-exclude-draft.patch`、`0311-megamoe-allow-unsupported-shape.patch`
（这些是 patch，不是 env overlay；需要单独验证 + 数值回归，请与 patch 负责人串行排期）。

---

## 6. 文件清单（本次新增，未覆盖任何既有文件）

`scripts/experiments/envs/`：
`exp51_draft_buckets.env`、`exp52_engram_overlap.env`、`exp53_limit_core_m20a28.env`、
`exp54_limit_core_aux16.env`、`exp55_kv_util_093.env`、`exp56_kv_util_095.env`、
`exp57_hccl_buffs_512.env`、`exp58_hccl_buffs_1024_explicit.env`、`exp59_hccl_buffs_2048.env`、
`exp60_taskq_lockin.env`、`exp61_malloc_conf_jemalloc.env`、`exp62_draft_probabilistic.env`、
`exp63_dp_device_ar.env`、`exp64_mc2_fullmesh_v2.env`、`exp65_kvpool_off_recheck.env`、
`exp66_long_prefill_4k.env`。

每个文件头部都写了：机理 / 预期 / 风险 / 验证口径 / 源码依据（file:line）/ 用法一行命令。
