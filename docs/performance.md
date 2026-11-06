# 性能与调优

所有数据来自单机 8×Ascend 910C（A3，16 die），配置：
DP2/TP8、Model Runner V2、DSpark K=5、FULL_DECODE_ONLY、1M 上下文上限、
`--max-num-batched-tokens 16384`、6 个优化 gate、
`HCCL_OP_EXPANSION_MODE=AIV`、jemalloc、external LB + prefix router。

> **数据批次**：下面的 2K/8K 表在 **2026-11-05** 于当前生产实例（连续运行约
> 2.5 小时、外部 KV 池 `host_shm` 开启）用生产 **16384** 批次、external LB +
> prefix router 采集；压测窗口内 serve 日志无其它在线请求。2K 分两组：单前缀
> （热前缀，`bench_2k.py`）和双前缀 A/B（`bench_dual.py`，两个长前缀分别落在
> 两个 DP rank）。router 对相同前缀做亲和，**单前缀低并发可能只压到一个 rank，
> 不能当作 DP2 上限**。8K 的 `bench_8k.py` 聚合值是 `单流 × 并发` 估算；
> 8K→1024 双前缀表是 wall-clock 实测。
>
> 历史 2026-11-02 / 2026-11-04 数据仍保留在本文档和 README 的批次 A/B 段落中，
> 但**不同实例、不同 KV 池状态之间不能直接相减**；容量规划以最新一批为准。

---

## 1. 2K prompt / 256 output

接口：`/v1/completions`，temperature=0，流式，开启 usage；
**2026-11-05**，16384 批次 + external LB/router，外部 KV 池开启，每个并发
重复 3 次取中位数。

### 1.1 单前缀（热前缀）

同一个 ≈2K prompt 发给所有请求；router 会优先命中已有 KV 前缀的 rank。

| 并发 | 总吞吐 (tok/s) | 单流 (tok/s) | TTFT (ms) |
|---:|---:|---:|---:|
| 1 | 87.6 | 98.4 | 308 |
| 2 | 186.1 | 104.7 | 295 |
| 4 | 340.0 | 95.8 | 305 |
| 8 | 645.1 | 91.9 | 336 |
| 16 | 1101.9 | 80.8 | 618 |
| 32 | 1834.6 | 73.7 | 667 |

```bash
python3 scripts/bench_2k.py --concurrency 1,2,4,8,16,32 --max-tokens 256 --repeats 3
```

### 1.2 双前缀 A/B（DP2 均衡）

两个互不共享首块的长前缀交替发送，两个 rank 各持有一个前缀，更接近多租户 /
多 system prompt 的并发场景。

| 并发 | 总吞吐 (tok/s) | 单流 (tok/s) | TTFT (ms) |
|---:|---:|---:|---:|
| 2 | 177.9 | 107.6 | 479 |
| 4 | 333.2 | 100.8 | 515 |
| 8 | 729.3 | 108.7 | 379 |
| 16 | 1292.9 | 94.2 | 405 |
| 32 | 1938.9 | 69.4 | 460 |
| 64 | 2009.8 | 54.9 | 671 |
| 64 持续×12 波 | 2009.8 | 54.9 | 671 |

```bash
python3 scripts/bench_dual.py --concurrency 2,4,8,16,32,64 --max-tokens 256 --repeats 3
```

观察：

- 单前缀与双前缀 c32 分别为 1,834.6 和 **1,938.9 tok/s**；双前缀在 c64
  仍能到 2,009.8 tok/s，但本次实例明显低于 2026-11-04 的 3.1–3.2k，
  且当前实例已运行约 2.5 小时并开启外部 KV 池，**不能与重启前实例直接对比**。
- 单前缀 c16/c32 的 TTFT 为 618/667 ms，双前缀 c16/c32 为 405/460 ms；
  双前缀更接近 DP2 均衡负载，容量规划应优先参考它。
- 单流吞吐在 c1–c8 最高（约 92–109 tok/s），c64 降到约 55 tok/s，是
  吞吐/时延的正常取舍；`--max-num-seqs 32` 与图桶上限对齐。
- 历史 2026-11-02 / 2026-11-04 的单前缀 C=32 为 1,832 / 1,933、双前缀
  C=32 为 2,253 / 2,124 tok/s；本次单前缀与前一批一致，双前缀偏低，
  需要在重启后的干净实例上复测后再决定是否调整 router / KV 池。

---

## 2. 8K prompt / 推测解码

8K 场景使用 `scripts/bench_8k.py`，脚本会：

1. 先发一轮 warm-up，让 `/metrics` 产生推测解码统计；
2. 从 `spec_decode_num_accepted_tokens_total` 与
   `spec_decode_num_draft_tokens_total` 计算接受率；
3. 换算 `tokens_per_step = 1 + acceptance_rate × K`；
4. 用 decode 时间 / 前向次数得到每步时延；
5. 输出每步 ms、单流 tok/s、聚合 tok/s、TTFT。

### 2.1 单前缀（`bench_8k.py` 估算）

```bash
python3 scripts/bench_8k.py --concurrency 1,4,8,16
```

| 并发 | 512 总吞吐 | 512 单流 | 512 TTFT | 1024 总吞吐 | 1024 单流 | 1024 TTFT |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 106.1 | 106.1 | 322 ms | 105.8 | 105.8 | 309 ms |
| 4 | 385.3 | 96.3 | 621 ms | 411.0 | 102.7 | 619 ms |
| 8 | 664.1 | 83.0 | 629 ms | 842.8 | 105.4 | 615 ms |
| 16 | 1242.2 | 77.6 | 678 ms | 1452.4 | 90.8 | 629 ms |

- 接受率：512 输出 **51.2–51.3%**，1024 输出 **51.6%**；tokens/step 分别约
  **3.56–3.57** 和 **3.58**。
- 脚本的“聚合”是 `单流 × 并发` 的估算值，不是 wall-clock 总 token 吞吐。

### 2.2 双前缀 8K→1024（wall-clock 实测）

```bash
python3 scripts/bench_dual.py --prompt-repeat 11 --max-tokens 1024 \
  --concurrency 2,4,8,16 --repeats 3
```

| 并发 | 总吞吐 (tok/s) | 单流 (tok/s) | TTFT (ms) |
|---:|---:|---:|---:|
| 2 | 210.7 | 110.9 | 324 |
| 4 | 403.9 | 106.3 | 366 |
| 8 | 764.8 | 103.1 | 648 |
| 16 | **1392.0** | 92.0 | 417 |

> 8K 数据的绝对值强依赖接受率抽签与 batch 组成，**不同进程之间的聚合吞吐不可直接
> 对比**。比较优化收益时请固定 prompt、temperature=0、并发，并优先看**每步时延**。
> 2.1 与 2.2 的聚合口径不同，不要直接横向比较；双前缀 2.2 更适合评估 DP2 实际
> 并发能力。

---

## 3. 关键优化的实测收益

| 优化 | 收益（参考） | 说明 |
|---|---|---|
| Model Runner V2（`VLLM_USE_V2_MODEL_RUNNER=1`） | 单流 +36%，并发 32 +68% | 相对 V1；需要补丁与 V2 兼容 |
| `HCCL_OP_EXPANSION_MODE=AIV` | 每步 −10.8% | 通信展开从 HOST 下沉到 Vector Core |
| jemalloc `LD_PRELOAD` | 每步 −3.2%～5.1% | 分配器争用下降 |
| PGO 版 libpython（可选） | 每步 −3.2%～5.1% | 必须在目标机编译，本仓库不含产物 |
| 6 个 `VLLM_ASCEND_V41_*` gate | 合计数个百分点 | 槽位映射标量化、QLI mode 3、Engram WKV TP 分片等 |
| DSpark K=5 | 显著提升单流 | 训练即 5-token 块；K 增大反而更慢 |
| `FULL_DECODE_ONLY` + 6N 桶 | 避免 draft 回退 eager | 缺 6N 桶收益大幅缩水 |
| `multistream_overlap_shared_expert` | 每步 −1.3%～4.6% | 共享专家多流；与 fused MC2 互斥 |

> 百分比随负载、batch 形状与接受率波动，只用于判断方向，不应作为容量规划的唯一依据。

---

## 4. 参数取舍

### 上下文长度

`--max-model-len` 直接决定 KV cache 上限。当前 1M + 16384 批次时每 die
KV ≈6.7 GiB（1,268,538 tokens/DP rank）；8192 批次时约 9.7 GiB
（2,091,049 tokens/DP rank）。缩短上下文可以换取更多并发余量，但会进入编译
缓存 key（改了就冷启动）：

```bash
DSV41_MAX_MODEL_LEN=262144 bash scripts/serve.sh
```

### 并发与批大小

| 参数 | 调大 | 调小 |
|---|---|---|
| `--max-num-seqs` | 聚合吞吐↑，单流时延↑，图桶需求↑ | 单流时延↓，吞吐↓ |
| `--max-num-batched-tokens` | prefill 吞吐↑，TTFT 抖动↑，激活峰值↑、KV↓ | TTFT 更稳，长 prompt 吞吐↓，KV↑ |
| `--gpu-memory-utilization` | KV 容量↑，OOM 风险↑ | 更安全，容量↓ |

`max-num-seqs` 与 `max-num-batched-tokens` 都参与调度器 hash，修改等于冷启动。

#### 8192 → 16384（实测容量）

| `max-num-batched-tokens` | 峰值激活 | KV cache | KV tokens / DP rank | 1M 上下文并发 |
|---:|---:|---:|---:|---:|
| 8192 | ≈4.1 GiB | ≈9.7 GiB | 2,091,049 | 1.99× |
| **16384（生产）** | **≈8.2 GiB** | **≈6.7 GiB** | **1,268,538** | **1.21×** |
| 32768（未采用） | ≈16.4 GiB（线性外推） | 预算为负 | < 1M | 无法满足 1M 上下文 |

32768 要让总占用回到 55.15 GiB 预算内，必须把 `max-model-len` 降到约 512K；
如果坚持 1M 上下文，**不要开 32768**。16384 是当前 1M 上下文下能保住的
最大批次。

### 图桶

默认 37 桶：

```
1,2,4,6,8,12,16,18,24,30,32,40,48,54,56,64,72,78,80,88,96,102,104,112,
120,126,128,136,144,150,152,160,168,174,176,184,192
```

- 6 的倍数保证 `num_reqs × (1+K)` 命中；
- 桶越密，图捕获时间与显存开销越高；
- 桶越少，越多 batch 形状退回 eager，decode 变慢。

### 推测解码

```bash
DSV41_SPEC_TOKENS=5     # 默认；DSpark 训练块大小
```

更大的 K 会增加 draft 开销；若接受率不足，净收益可能为负。
通过 `/metrics` 的接受率判断是否需要调整。

---

## 5. 测量注意事项

1. **预热**：首个请求包含算子编译/图捕获，必须丢弃。
2. **温度 0**：降低采样随机性，但 MoE 路由与并发调度仍会带来波动。
3. **不要跨进程比吞吐**：接受率抽签差异会盖过优化收益；优先比较每步时延。
4. **固定 prompt**：prefix cache 命中会显著改变 TTFT；压测前清理或统一预热。
5. **观察显存**：`npu-smi info` 与日志中的 `Free memory on device` 行可用于判断
   是否触及 HBM 上限。
6. **NFS 抖动**：权重在 NFS 上时，首次加载与页回收会引入额外方差。
7. **router 前缀亲和**：相同 prompt 会被粘到已有 KV 前缀的 rank；测 DP2 并发上限
   要用 `bench_dual.py` 的双前缀，或人为制造两个/多个长前缀，否则低并发可能只
   压到一个 rank。
8. **聚合口径**：`bench_8k.py` 的 aggregate 是 `单流 × 并发` 估算；
   `bench_dual.py` 按实际总 token / wall time 统计，跨脚本不要直接比吞吐。

---

## 6. 容量规划参考

| 指标 | 数值 |
|---|---|
| 单 die HBM | 64 GB（可用 ≈61.3 GiB） |
| 权重 / die | ≈38.3 GiB |
| KV cache / die @1M + 16384 批次 | ≈6.7 GiB（1,268,538 tokens/DP rank） |
| KV cache / die @1M + 8192 批次 | ≈9.7 GiB（2,091,049 tokens/DP rank） |
| 峰值激活 @16384 | ≈8.2 GiB |
| 峰值激活 @8192 | ≈4.1 GiB |
| 非 torch + 图 | ≈3.8–4.0 GiB |
| 典型整机聚合吞吐 | ≈2.25k tok/s @ 并发 32 / 2K 双前缀（实测）；≈1.34k @ 并发 16 / 8K→1024 双前缀（实测） |

2K 单热前缀 @c32 约 1.83k tok/s；双前缀约 2.25k，说明热前缀亲和会牺牲部分
DP2 利用率。需要更高并发时，优先评估缩短 `max-model-len` 或增加节点，而不是继续抬高
`gpu-memory-utilization`——后者会把 OOM 风险转成运行时抖动。

---

## 7. 生产日志分析：DP lockstep 与 prefill 停顿

上面的基准都是短 prompt、少量前缀。真实长上下文负载下，总输出 tok/s 的大幅波动主要不是
router 或 prefix cache 的问题。数据来自 2026-11-04 08:16–12:26 的 `serve_*.log`
（约 2100 个请求，平均 prompt 约 66–84K token，`--max-num-batched-tokens 16384`，
`--max-num-seqs 32`，启动日志 `NPU KV cache size: 848,184 tokens`/DP rank），用
`scripts/analyze_serve_log.py` 复现：

```bash
python3 scripts/analyze_serve_log.py logs/serve_*.log                       # 全程
python3 scripts/analyze_serve_log.py logs/serve_*.log --since 09:40 --until 10:40
```

| 指标（10 秒窗口，两个 rank 都在跑 ≥5 个请求） | 数值 |
|---|---|
| 总输出 tok/s：均值 / 中位数 / p95 | 158 / 30 / 695 |
| 两个 rank 同时“停顿”（各 ≤40 tok/s）的窗口占比 | **65.6%** |
| 两个 rank 同时“健康”（各 ≥150 tok/s）的窗口占比（此时合计均值） | 19.0%（590 tok/s） |
| 一个 rank 停顿时另一个也停顿的概率（对比无条件概率） | 0.98（0.66） |
| 两个 rank 输出 tok/s 的相关系数 | 0.95 |
| 单步耗时（由推测解码计数估算）：健康 / 停顿 | ≈107 ms / ≈2.5 s |
| 有 rank 在做 prefill（KV 10 秒内增长 ≥1%）时的停顿占比 / 合计 tok/s | 78.1% / 110 |
| 没有 prefill（KV 基本不涨）时 | 31.0% / 285 |
| 停顿持续时间：中位 / p90 / 最长 | 30 s / 90 s / 280 s |

**机制。** DP2 + 专家并行下，两个 EngineCore 每一步都要同步（MoE 的 all-to-all 跨越 DP×TP；
vLLM `DPEngineCoreProc.run_busy_loop` 每步做全局同步，空闲 rank 也要跑 dummy batch）。
一个 rank 处理 16384 token 的 prefill 分块时，这一步要 ~2.5 s，另一个 rank 的 decode 也
只能跟着每 2.5 s 走一步，于是约 10 个正在 decode 的请求每秒只产出 ~1 token/请求。
这解释了为什么 router 把缓存命中率做到 81–91% 之后总吞吐仍然被 prefill 主导。

**另一个独立的问题：一个 rank KV 被打满。** 09:40–10:40 rank0 的 KV 平均 86–87%
（p90 约 97%）并有约 3 个请求排队，rank1 只有 46–58%、没有排队；rank0 每 20 分钟重算
约 6M 未命中 token，rank1 约 1M，rank0 命中率从 84% 掉到 58%，输出 tok/s 减半。
两个 rank 的 active 请求数只差 1–2 个。这是旧版按请求数均衡的 router 看不到的，
新版 `cost` 策略（`docs/router.md` 3.2）用 KV 使用率 / 排队深度 / prefill 排队 token 来避开。

**router 能做什么、不能做什么。** 能：不让某个 rank 被打满、把冷请求的 prefill 摊到两个
rank（两个 rank 同时 prefill 时，一个停顿窗口推进的 prompt KV 约为只有一个 rank prefill
时的 2.6 倍，而 decode 输出同样被拖慢）。不能：消除 lockstep 本身。若这些都不够，剩下的是
架构层面的选择（例如单引擎 TP16、或 prefill/decode 分离），不在本仓库范围。

### 批次从 16384 改为 8192 的验证方法（可选）

停顿步的时长与 prefill 分块大小成正比；同时 8192 比 16384 少占约 4 GiB/die 激活
（见第 4 节），其中大部分变成 KV（第 4 节实测 6.7 → 9.7 GiB/die，1M 上下文口径）。
上面那次日志里 KV（848K token/rank，约 10 个请求）是并发上限，因此可以尝试 8192。
**这是依据日志机制提出的建议，尚未在 8192 下实测**——预期是停顿步变短、KV 更宽松，代价是
prefill 效率下降、大块未命中 prompt 的 TTFT 变长。

```bash
# config/dsv41.local.env  —  可选对比
DSV41_MAX_NUM_BATCHED_TOKENS=8192     # 改动会进入编译缓存 key → 冷启动
```

判据：用 `analyze_serve_log.py` 比较“两 rank 同时停顿占比”“总输出 tok/s 均值 / p5”
“KV p90 与排队”，同时看客户端侧 TTFT p95，并核对启动日志的 `NPU KV cache size`
（16384 时是 848,184 token/rank）。每组至少跑 60 分钟——上面那次 KV 打满持续了约一小时。
如果同时升级了 router 策略，想分开归因，就让第一组仍用 16384、第二组再改为 8192。
**注意**：已有的 `config/dsv41.local.env` / `config/dsv41.env` 里的显式值会覆盖默认值。

**口径提醒。** “停顿”阈值（≤40 / ≥150 tok/s）和“KV 增长 ≈ prefill”都是代理指标，
`min-running` 等参数可在脚本里调；表中数字用于判断方向，不是容量规划值。
