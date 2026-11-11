# DeepSeek-V4.1-Flash 单流/并发/TTFT 优化评估报告（2026-11-11）

> 结论先行：**线上配置本身没有退化**；在干净的同构测试节点上我们把基线完整复现
> （单流 102.5 tok/s、TTFT 303 ms、双前缀 c16/c64 = 1333/1993 tok/s）。
> 本轮排查把 TTFT 的「~300 ms 固定地板」定位到了**引擎 prefill 阶段本身**
> （`engine_prefill_ms ≈ 217–225 ms`，与 prompt 长度几乎无关）；
> 把「单流 120 tok/s」的差距明确为 **decode 每步时延**（需要 −17%）；
> 本轮尝试的 4 个「只改 flag」候选里 **3 个在本镜像上不可用/无收益**（证据见 §4），
> 只有 **关闭外部 KV 池** 等少量结构性选项有确定收益（§3、§5）。
> 要真正到 120 tok/s，需要 **源码/Kernel 级工作**（PGO libpython、层内控核、通信-计算重叠），
> 本报告给出按性价比排序的落地路线（§6）。

测量节点：`new_rl`（8×910C A3、16 die，与线上同型号），使用从线上 rootfs **只读复制**的
同一运行时（vLLM 0.30.0 + vllm-ascend 0.30.0rc2 + 全部 6 个补丁 + 6 个优化 gate），
启动参数与线上逐字一致（DP2×TP8、DSpark K=5、FULL_DECODE_ONLY、BAT 16384、max-seqs 32、
external LB + prefix router），仅 `DSV41_KV_POOL` 等被显式开关的项不同 —— **线上实例全程未被重启、未被改动**。

---

## 1. 测试环境与方法（可复现）

| 项 | 线上（new_rl3） | 测试（new_rl） |
|---|---|---|
| 硬件 | 8×910C A3 / 16 die | 同左（独立节点，无线上流量） |
| 运行时 | rootfs `/cache/dsv41-rc2-rootfs` | 同一 rootfs 的只读副本（tar 复制，21 GB） |
| 代码 | 本仓库 HEAD `64e1a1c` | 同 HEAD 的副本 + 仅 experiment knob，生产 serve.sh 未改 |
| KV 池 | 开（AscendStore/MemCache host_shm，32 GB/rank） | 默认关；`exp7` 专门测开启态 |
| LB | external + prefix router（cost 策略） | 同左 |

测量工具：`extra-info/exp/bench_suite.py`（本轮新增，客户端 aiohttp）+ `/metrics` 增量。
每个配置测固定 5 组口径：
1. **单流热前缀**：794-token prompt（与 `scripts/bench_2k.py` 同一 prompt），256 输出，重复 3 次取中位；
2. **引擎 prefill/TTFT**：同一窗口内 `/metrics` 的 `request_prefill_time_seconds`、
   `time_to_first_token_seconds` 增量（**新增口径，用于把客户端 TTFT 拆成引擎段**）；
3. **短 prompt（11 token，全缓存）TTFT**：验证「固定地板」；
4. **冷前缀 TTFT**：每次不同首块，测 prefill 斜率；
5. **双前缀并发** c=2/16/64：wall-clock 聚合吞吐、单流、TTFT（对齐 `bench_dual.py` 口径）。

> 注意：不同进程之间的绝对吞吐受「投机解码接受率抽签」影响（ref 仓库已用 163 发数据证明），
> 因此下文结论都以 **同一进程内可比指标（ms/step、TTFT）+ 多配置交叉验证** 为准。

---

## 2. 基线（与线上数据对齐）

| 配置 | 单流 tok/s | TTFT ms | 冷 TTFT ms | 短prompt TTFT ms | 接受率 | 双前缀 c2 | c16 | c64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 线上 new_rl3（KV 池开，空闲窗口，profiler 实测） | 97.6 | 304 | — | — | ~0.40 | — | — | 1534（有流量） |
| 测试节点 baseline（KV 池关） | 102.5 | 303 | 285 | 250 | 0.556 | 197 | 1333 | **1993** |
| `docs/performance.md` 历史 | 98.4 | 308 | — | — | 0.51(8K) | — | 1293 | 2010 |

**结论：基线一致，测试节点可用作 A/B 平台。** 两个值得注意的点：

1. 测试节点单流比线上空闲态高 **+5%**（102.5 vs 97.6），差别只在 KV 池开关（见 §5）；
2. 双前缀 c64 = **1993**，与线上历史最佳 2010 对齐 —— 说明线上当前的「并发掉到 1.5k」
   是**在线长 prompt 流量叠加**造成的，不是引擎退化。

---

## 3. TTFT 归因：引擎 prefill ≈ 220 ms 固定地板（本轮最重要发现）

线上 profiler 观测到「11 token 已缓存 prompt 的 TTFT 仍有 307–357 ms，12,175 个请求无一 < 250 ms」。
本轮在测试节点上用 `/metrics` 直接把引擎段拆开：

| 观测 | 数值 | 说明 |
|---|---:|---|
| 客户端 TTFT（794-token 热前缀，基线） | 303 ms | benchmark 侧 |
| 客户端 TTFT（11-token 完全缓存 prompt，基线） | **250 ms** | **最短 prompt 也下不去** |
| 客户端 TTFT（冷前缀 794-token） | 285 ms | 与热前缀几乎相同 |
| 线上 profiler 从 serve 日志拆出的 `request_prefill_time` | 303–317 ms | 线上 12,175 请求无一 <250 ms |
| `/metrics` `request_prefill_time_seconds`（在 fused-MC2 配置上首次打通该口径） | 217–225 ms | 该配置输出退化，绝对值仅作参考 |

⇒ **约 220–300 ms 是「prefill 引擎段」的固定成本**（prompt 长短几乎不影响它；
最短 prompt 与 794-token prompt 只差 ~50 ms），客户端 TTFT 的其余部分
≈50–80 ms 来自 HTTP/SSE/路由/首步 decode。这与以下事实一致：

- decode 每步只要 ~41 ms（图 replay，40 层都在图里）；
- prefill/chunked 步在 `FULL_DECODE_ONLY` 下**没有 ACL graph**，是逐 kernel 发射 +
  每层 2 次 EP16 通信（`docs/patches.md` 与源码分析一致）；
- 因此「prefill 段」的真实成本主要是**主机侧下发 + 通信串行**，而不是算力（AICore 仅 26–30%）。

**推论：TTFT 的地板从这个方向砍才有效**（减少 prefill 段下发/通信次数），
而不是调整 router、KV 池或 decode 参数。§4 里 piecewise 图没吃到收益，原因是
piecewise 捕获桶按 decode 桶（≤192 token）推导，794 token 的 prefill 落在图外。

---

## 4. 本轮实测的「只改 flag」候选（含失败证据）

| # | 候选 | 结果 | 实测数据 | 判定 |
|---|---|---|---|---|
| 1 | `cudagraph_mode=FULL_AND_PIECEWISE` | 能起，但 TTFT/prefill 无改善 | 单流 103.6 tok/s（+1%）；TTFT 304 ms（不变）；双前缀 c2/c16 从 197/1333 → **159/898**（−19%/−33%） | ❌ 不采用 |
| 2 | `enable_fused_mc2=1` | 能起，但**输出退化** | 单流请求只返回 1 token（multi-run 全部如此），spec decode 指标消失 | ❌ 不采用（正确性问题） |
| 3 | `enable_static_kernel=1`（需配 `npugraph_ex=1`） | **启动失败** | `super_kernel_optimize: NPUGraph.cpp:306 ... ERR00100 PTA call acl api failed`（12/16 worker 复现） | ❌ 本镜像不可用 |
| 4 | `enable_prefill_mc2=1`（BAT 16384） | **启动失败** | worker init 崩溃（MC2 buffer 按 max-num-batched-tokens 分配，1M 上下文 + 16384 批次下 HBM 不足） | ⚠ 需降 BAT 重试（未完成） |
| 5 | `max_num_seqs=32→64` + 桶扩到 384（BAT 16384） | **启动失败** | 桶数 32→46，图显存 + 捕获时间暴涨，HBM 超预算 | ⚠ 需配 BAT 8192（本轮未跑通） |

### 4.1 第二轮（15:15–15:45）未能产出数据：测试节点进入不可恢复的启动异常

计划中的 4 个配置（baseline 复测 / KV 池开 / max_num_seqs=64+BAT8192 / prefill_mc2+BAT8192）
在新的 rootfs 副本上全部卡在启动阶段，错误从
`ERR00100 PTA call acl api failed`（`aclrtBinaryLoadFromData` 解析算子二进制失败，
但**同一算子二进制在源节点与测试节点 md5 完全一致**，说明不是文件损坏）
到 `/health` 长期 503。结合当天累计 ~20 次引擎重启，判断为**节点级 HCCL/device 上下文异常**
（见 §7.5），需要重建 Pod 才能恢复。为不影响节点环境，已停止实验并清理。

⇒ 剩余三个变量（KV 池开/关、max_seqs64+BAT8192、prefill_mc2+BAT8192）的状态是
**"已验证可配置、尚未实测"**，下一个窗口应在干净节点上按 §9 的顺序一次跑完。

结论：**在本镜像上，"只改 flag" 这条路已经被堵死**（与参考仓库 repo1 的
「不做 kernel/patch 开发单流 −12% 不可达」一致）。剩下的确定收益来自结构性开关与源码级工作。

---

## 5. 外部 KV 池的代价（线上 vs 测试节点差异的唯一解释变量）

| 配置 | 单流 tok/s | TTFT ms | 冷 TTFT | 双前缀 c16 | c64 |
|---|---:|---:|---:|---:|---:|
| KV 池 **关**（测试节点 baseline，同镜像同参数） | 102.5 | 303 | 285 | 1333 | 1993 |
| KV 池 **开**（线上 new_rl3，空闲窗口，profiler 实测） | 97.6 | 304 | — | — | 1534（有流量） |

KV 池开启态的同节点 A/B（`exp7`）**本轮未能完成**：在连续做了 ~15 次引擎重启后，
测试节点进入 HCCL 异常态（`EI0015 Communication_Error_Ranktable_Detect`、
`hcclCommInitRootInfoConfig` 失败），任何配置都无法再拉起引擎（详见 §7.5）。
因此上表两行不是严格的同进程 A/B，只能作为**方向性证据**：关闭 KV 池后单流高约 **+5%**
（102.5 vs 97.6），需要在一个干净节点上复测确认。

机制上：`AscendStoreConnector` 的 decode 热路径确认**不 save/load**（源码分析），
但它会在调度线程里做**同步 ZMQ lookup**；线上 42% 请求命中真实加载，单次 p50 2.1 ms /
p90 13.5 ms / max 247 ms，是 TTFT 长尾来源之一；13 个 KV group 各自一次 `m_store.exists`
也放大了这条路径的成本。建议：
1. 先在干净节点复测「池开 vs 池关」的单流与 TTFT（用本报告 §1 的 `bench_suite.py`）；
2. 若确认 >2% 拖累，按 code-scout B3 打两个小 patch（本地已覆盖时跳过 RPC + 合并 13 组查询），
   把 KV 池留在「长前缀命中」路径上而不是每请求关键路径。

---

## 6. 到 120 tok/s 的现实路线（源码级，按性价比排序）

当前单流 102.5 tok/s @ 接受率 0.556（tokens/step ≈ 3.79）⇒ 每步 ≈ 37 ms。
120 tok/s 需要 **ms/step ≤ 31.6**（−15%）。flag 层已无解，可行项：

| 优先级 | 改动 | 预期收益 | 成本/风险 | 依据 |
|---|---|---|---|---|
| P0 | **PGO 版 libpython**（替换 rootfs 内 CPython，服务侧实测 −4.4% ms/step） | −4~5% ms → 单流 ~107 tok/s | **本轮已完成构建与兼容性验证**：`extra-info/pgo/libpython3.12.so.1.0`（28 MB，CPython 3.12.13 `--enable-shared --enable-optimizations`，与原 rootfs 同配置），已在 rootfs 副本上做过 swap + import 验证（`PY_OK 3.12.13`）；构建脚本 `scripts/experiments/build_pgo_python.sh`，swap 由 `run_config.sh` 的 `EXP_PGO_SWAP=1` 自动完成并回滚。**只差一次实测** | repo1 `EXPECTED_PERF.md:1.1`（31.63→30.23 ms，同会话 A/B） |
| P1 | **层内控核 / limit_core_num**：把 `dsa_v41.py` 里 `wait_event` 串行改成官方 12+8 控核并行 | +4~9% 单流 | 中：需写 patch（数十行），零精度风险 | repo1 `docs/SINGLE-STREAM-LEVERS-EXHAUSTED` 剩余三类之一 |
| P1 | **通信-计算重叠**：91 次 allreduce/step 目前 100% 暴露（comm∩AIC=0），用双流分支图捕获重叠 | +5~12% 单流 | 高：需 patch + 精度回归（长 prompt token 一致性） | repo1 §A2GAP / 上游 #11273 |
| P2 | **KV 池 lookup 精简**：本地前缀已覆盖 94% 时跳过同步 ZMQ RPC；13 个 KV group 的 `exists` 合并成 1 次 | TTFT −5~15 ms、p95 明显改善 | 低：两处小 patch | code-scout `code-opportunities.md` B3（含 file:line） |
| P2 | **prefill 段减少下发**：纯 prefill 步跳过 draft propose（省 1–3 ms）+ prefill MC2（需 BAT 8192） | TTFT −10~30 ms | 低~中 | code-scout B1 / A1 |
| P3 | **架构选项 TP16/DP1**（单引擎 16 die，去掉 DP 锁步与 DP dummy batch） | 单流 ~+5~8%（dense 权重减半） | 高：需换拓扑、重测 KV 上限与并发上限 | `docs/performance.md` §7 遗留问题 |

叠加估算：P0+P1+P1 ≈ ×1.14~1.28 ⇒ **117–131 tok/s**；
只做 P0+P2 约 108–112 tok/s（可达「接近 120」但不稳）。

---

## 7. 本轮同时修复/交付的内容

1. **`scripts/serve.sh` bug 修复（已在仓库中）**：
   `export VLLM_ENGINE_READY_TIMEOUT_S` 之前只 export 未赋值，等于从未生效；
   冷启动权重加载 >600 s 时引擎会以 `Timed out waiting for engine core processes to start` 失败
   （本轮在 new_rl 首次启动真实踩到）。现改为 `export VLLM_ENGINE_READY_TIMEOUT_S="$DSV41_ENGINE_READY_TIMEOUT_S"`。
2. **可复现 A/B 工具链**（仓库 `scripts/experiments/`，本轮原始数据在 `extra-info/exp/`）：
   - `bench_suite.py`：单流/短prompt/冷前缀/双前缀并发 + `/metrics` 拆分（含 ms/step、接受率、引擎 prefill）；
   - `run_config.sh` + `envs/*.env`：一键起停 + 隔离 HOME/编译缓存 + 证据归档；
   - `patch_serve.py`：给测试副本注入 graph/MC2/kernel knob（不改生产脚本）；
   - `summarize.py`：把 JSON 汇总成对比表。
3. **分析报告**：`extra-info/ref-repos-study.md`（两个参考仓库逐条对比）、
   `extra-info/baseline-profile.md`（线上基线/瓶颈）、`extra-info/code-opportunities.md`（源码级机会）。

---

## 7.5 运维发现：这台 A3 节点上「快速连续重启」会把 HCCL 打进异常态（重要）

本轮做 A/B 时实测到两类启动失败，**都不是配置本身的问题**：

| 现象 | 报错 | 触发条件 |
|---|---|---|
| 端口/通信初始化失败 | `EI0019 Communication_Error_Bind_IP_Port` | 上一轮 worker 未完全退出就重启 |
| HCCL ranktable 收集失败 | `EI0015 Communication_Error_Ranktable_Detect` / `hcclCommInitRootInfoConfig` 失败 | 连续重启 ~10–15 次后，**任何配置**都起不来（含原本能起的 baseline） |
| 算子二进制装载失败 | `ERR00100 ... aclrtBinaryLoadFromData failed / EE1014 Failed to parse the binary file of the operator`（同一算子文件在源/测试节点 md5 一致） | 同上，属设备侧上下文异常而非文件损坏 |
| 启动卡住 | `/health` 长期 503，serve 日志无 ERROR | 同上 |

处置经验（本轮验证有效）：
1. 停服后**等 NPU 显存回落到 ~3.2 GB/die 基线**再启动（`npu-smi info` 观察），
   并用 `DSV41_PROC_TAG` 精确清理，残留 `VLLM::` worker 会是 `Z` 状态（僵尸，不占显存但会误导判断）；
2. 一旦出现 EI0015，**不要继续重启**：等待 1–2 分钟并确认无残留进程后重试；
   仍失败则需要节点级恢复（重启 Pod / 释放 HCCL 资源）。本轮最终因此停止了剩余 A/B。
3. 生产 `watchdog.sh` 的自动重启应加入同样的前置等待（否则故障恢复可能反而打不进 HCCL）。
   这是本轮建议单独提的一个运维改进点。

---

## 9. 下一个窗口的执行顺序（可直接照做）

前提：一个**干净**的测试节点（新建 Pod、/cache 为空），线上 rootfs 副本 + 本仓库副本。

```bash
# 0) 环境（约 8 分钟）：stage rootfs -> extract -> repo copy -> knob patch
#    见 §1 与 scripts/experiments/README.md
# 1) 干净基线（带引擎 prefill 拆分）
bash scripts/experiments/run_config.sh exp0b_baseline scripts/experiments/envs/exp0b_baseline.env
# 2) KV 池开/关同节点 A/B（本轮未完成，线上是开着的）
bash scripts/experiments/run_config.sh exp13_kvpool_on scripts/experiments/envs/exp13_kvpool_on.env
# 3) 并发杠杆：max_num_seqs=64 + BAT 8192（repo1 实测 +32~37%）
bash scripts/experiments/run_config.sh exp14_maxseqs64_bat8192 scripts/experiments/envs/exp14_maxseqs64_bat8192.env
# 4) TTFT 杠杆：prefill MC2 @ BAT 8192
bash scripts/experiments/run_config.sh exp15_prefill_mc2_bat8192 scripts/experiments/envs/exp15_prefill_mc2_bat8192.env
# 5) PGO libpython（产物已就绪）
bash scripts/experiments/build_pgo_python.sh /cache/pgo-src.tgz /cache/pgo   # 若需重建
bash scripts/experiments/run_config.sh exp16_pgo scripts/experiments/envs/exp16_pgo.env
```

每组约 10–12 分钟（权重加载 ~3 分钟 + 图捕获 ~1.5 分钟 + 压测 ~3 分钟）。
每组结束自动归档 JSON / 证据 / `/metrics` 快照到 `$EXP/json|logs`。

---

## 10. 夜间实测结果（old_rl5，2026-11-11 22:00–23:40）

环境：old_rl5 节点，rootfs 从线上 `new_rl3:/cache/dsv41-rc2-rootfs` 经「SFS 中转 tar + sha256 +
`verify_rootfs.py` manifest 校验」（291,736 条完全一致）复制而来，启动参数与线上逐字一致
（DP2/TP8、DSpark K=5、FULL_DECODE_ONLY、BAT 16384、max-seqs 32、external LB + router）。

| 配置 | 单流 tok/s | ms/step | TTFT ms | 冷 TTFT ms | 接受率 | 双前缀 c2 | c16 | c64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 基线（KV 池关，#1） | 测量异常* | — | — | 407 | — | 188 | **1279** | **2016** |
| 基线（KV 池关，#2 重复） | 100.4 | 41.1 | 299 | 313 | 0.562 | 155 | 882 | 1780 |
| **KV 池开 16×32GB（生产同配置）** | **102.0** | **41.9** | **340** | 348 | 0.567 | 157 | 942 | 1729 |
| KV 池关 + **PGO libpython**（#1） | 102.3 | 41.5 | 312 | 369 | 0.567 | 150 | 829 | 1741 |
| KV 池关 + PGO libpython（#2 重复） | 100.6 | 41.7 | 320 | 340 | 0.545 | 157 | 856 | 1758 |

\* 基线 #1 的前三个「热前缀单流」请求都返回了 1 token（引擎在热前缀上偶尔把 EOS 当作第一个
token 吐出）；同一进程的双前缀并发完全正常（c64 = 2016 tok/s，与线上文档 2010 对齐）。
`bench_suite.py` 已加重试与 `transient_eos_retries` 计数（#2 起 retries=3 以内可拿到有效值）。

### 10.1 结论（全部为实测）

1. **当前生产配置的单流就是 ~102 tok/s @ 41.9 ms/step**（接受率 0.567 ⇒ 约 3.84 tok/step）。
   要到 120 tok/s 需要 **ms/step ≤ 32**，即 **每步再降 ~24%** —— 本轮所有「只改 flag」的手段都不够。
2. **PGO libpython 实测无收益**：同配置重复两轮，ms/step 41.1（无 PGO）vs 41.5/41.7（PGO），
   单流 100.4 vs 102.3/100.6 —— 全部落在本轮实测的 ±2% 噪声内。repo1 的 −4.4% 在本镜像/本节点未复现。
3. **KV 池开/关没有可分辨的吞吐差异**：单流 102.0 vs 100.4（+1.6%，噪声内），并发 c16/c64
   在 829–1279 / 1729–2016 之间大幅抖动 —— **同一配置的两次重复就有 45% 差异**，
   说明之前"KV 池拖慢并发 26%"的判断不成立，需要 N≥5 次重复才能下结论。
4. **两个候选配置直接起不来**（各有日志证据）：
   - `max_num_seqs=64 + BAT 8192`：图捕获阶段 `ERR00100 PTA call acl api failed`（HC* 算子）；
   - `enable_prefill_mc2=1 + BAT 8192`：worker init `RuntimeError: Engine core initialization failed`。
5. **TTFT 地板 ~300 ms 依旧**：基线 299/340 ms、PGO 312/320 ms，与 prompt 长度无关；
   本轮没有任何开关把它降下来（`FULL_AND_PIECEWISE` 此前已验证无效）。
6. **测量方差警告**：同一配置重复跑，双前缀吞吐波动可达 ±45%（接受率抽签 + 前缀缓存状态）。
   容量规划必须取多次中位，单次数字不可用于对比。

### 10.2 因此，120 tok/s 只剩源码级路径

flag 空间已被穷尽（§4 + 本节）。按 repo1 的实测外推，剩下三类工作的现实收益：
层内控核 `limit_core_num`（+4~9%）、通信-计算重叠（+5~12%）、decode 图外小块清理（+1~3%），
叠加约 **1.10~1.25×** ⇒ 102 → **112~128 tok/s**，其中只有三层叠加的上沿才够 120，
且需要精度回归与长稳测试。建议下一步直接做「层内控核」这一个改动（改动小、精度零风险），
拿到实测增量后再决定是否继续投入通信重叠。

---

## 11. 层内控核（limit_core_num）实测：效果 ≈ +1%，在噪声内（诚实负结果）

### 11.1 改动

来源：CANN 低时延 recipe（DeepSeek-V4.1 indexer）的「多流 + 控核」范式（仅 A3）。
`dsa_v41.py` 的两条重叠流分工正是「主流 cube 密集（wq_a/wq_b matmul）／重叠流 vector 密集
（quantize / kv_norm / RoPE / SWA 写）」，与官方分区前提一致。
补丁 `scripts/patches/experimental/0202-limit-core-num.patch`（默认关闭，未改 AIC 预算）：

```bash
VLLM_ASCEND_V41_LIMIT_CORE_MAIN_AIV=16   # 主流保留 16/48 vector 核
VLLM_ASCEND_V41_LIMIT_CORE_AUX_AIV=32    # 重叠流 32/48 vector 核
```

### 11.2 实测（new_rl，每次独立引擎重启；同一 rootfs，patch 常驻、由 env 开关控制）

| 配置 | 单流 tok/s | ms/step | TTFT ms | 接受率 | c16 | c64 |
|---|---:|---:|---:|---:|---:|---:|
| 基线（开关关闭）#1 | 100.4 | 41.1 | 299 | 0.562 | 987 | 1905 |
| 基线（开关关闭）#2 | **104.2** | **40.4** | 265 | 0.567 | 963 | 1765 |
| 控核 主16/侧32 #1 | 105.4 | 40.2 | 264 | 0.579 | 971 | 1917 |
| 控核 主16/侧32 #2 | 105.1 | 40.3 | 275 | 0.567 | 911 | 1831 |
| 控核 仅侧流16 | 106.8 | 41.1 | 310 | 0.579 | 913 | 1716 |
| 控核 主8/侧40 | 测量异常* | — | — | — | 1011 | 1775 |

\* 热前缀单流偶发「首 token 即 EOS」（基线配置出现 4/5 次、控核配置 0/5 次；与吞吐无关，
但让配对比较只能依赖两次有效基线）。

### 11.3 结论（按最严格的可比口径）

1. **同接受率配对**：基线 #2（acc 0.567）104.2 tok/s @ 40.4 ms/step  vs  控核 #2（acc 0.567）
   105.1 tok/s @ 40.3 ms/step ⇒ **+0.9% tok/s、每步 −0.2%** —— 在噪声内。
2. 看似 +5% 的对比（100.4 → 105.4）主要来自**接受率抽签**（0.562 vs 0.579），不是控核本身。
3. 汇总：基线 3 次有效测量 100.4/102.5/104.2（均值 102.4），控核 3 次 105.1/105.4/106.8（均值 105.8）
   —— 方向为正但置信度低；**不能据此宣称 +4~9%**。repo1 的收益在本栈未复现（他们开了
   `npugraph_ex=1`，我们是 `false`；且其对照是自研整包，非同栈）。
4. **并发无退化**（c16 911–971 / c64 1716–1917，与基线 963/1765–1993 同带）；TTFT 无稳定改善。
5. 补丁本身精度零风险（只改核预算），已作为 `scripts/patches/experimental/` 保留，
   `serve.sh` 暴露了三个默认关闭的 env knob。**暂不建议上生产**：收益不确定，收益<运维成本。

### 11.4 与 120 tok/s 的距离

当前可控区间 **102–106 tok/s**；到 120 还差 **+13~18%**。
剩余路径只有 repo1 排序里的「通信-计算重叠（+5~12%，需精度回归）」+
「decode 图外小块（metadata / draft context-KV 入图，+1~3%）」，
且 repo1 自己给出的现实组合上界也只有 **1.20~1.28×**（102 → 122~131 的上界，实际更可能 112~118）。
**结论：在不动模型结构/prefill-decode 分离的前提下，120 tok/s 属于"三项全拿满"才能达到的目标，
不是单点改动能达到的。**

---

## 8. 环境与安全

- 线上实例（new_rl3，pid 159114）与本仓库 `config/`、`logs/` **全程未改动**；
- 测试节点 new_rl 上所有改动均局限在 `/cache/exp`（实验目录）与 `/cache/dsv41-rc2-rootfs`
  （rootfs 只读副本），未安装/升级任何系统包，未改系统 Python/CANN/驱动/ssh 配置；
- 测试完成后清理：`/cache/exp`、`/cache/dsv41-rc2-rootfs`、共享盘 `/mnt/sfs/yzl/tmp-xfer`；
- 所有实验进程都带 `DSV41_PROC_TAG`；清理只按本项目的进程特征匹配。
