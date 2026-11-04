# 外部 KV 池（Mooncake / AscendStoreConnector）

本文档描述本项目的外部 KV 池：把 prefix cache 的 KV block 从 NPU HBM 扩展到
**宿主内存**（Mooncake Store），让重复前缀（长文档、多轮对话、共享 system
prompt）在跨请求、跨 DP rank 之间命中，而不是重复 prefill。

> ⚠️ **本机当前状态（2026-11-05，A3 910C）：不可用，已通过
> `config/dsv41.local.env` 设为 `DSV41_KV_POOL=0`。**
> 跨卡 HIXL 传输在本机稳定失败（`halDeviceDisableP2P ret:17`），一旦命中
> 别的 rank 写入的 block 就会让 worker 退出，且崩溃后 HIXL 连初始化都失败。
> 代码默认仍是 `DSV41_KV_POOL=1`（便于驱动/环境修好后一键恢复），本机由
> local.env 关闭。复现证据、规避方式与恢复步骤见第 3.1 节。

参考实现：ModelScope `x0ken0/vllm-ascend-a2-dsv41f`（A2 / 910B3 / Docker）。
本项目在其基础上按 A3（8×910C、DP2×TP8、native rootfs）适配。

## 1. 作用与收益

| 维度 | 无外部 KV 池 | 有外部 KV 池（默认 32GB/卡） |
|---|---|---|
| KV 存放 | 仅 NPU HBM：1,268,538 tokens/rank（≈6.7 GiB） | HBM + 宿主内存：32 GiB/卡 × 16 卡 = **512 GiB** |
| 等效容量 | 1.27M tokens/rank | ≈ **5.8M tokens/卡（≈4.6×）** |
| 复用范围 | 单 rank 生命周期内 | 跨请求、跨 DP rank；同一 master 下可跨实例 |
| 长文档二次提问 | 重新 prefill（慢，且 DP lockstep 阻塞另一 rank） | 直接命中，几乎免 prefill |
| 重启后 | 空 | 空（除非启用 SSD offload） |

实测参考：KV 单价约 **5.5 KB/token**（6.7 GiB / 1,268,538）。
容量换算：`tokens ≈ 32 GiB / 5.5 KB ≈ 5.8M`（64GB/卡 时为 ≈11.6M，但在这台
2 TiB 节点上会因注册失败而无法启动，见第 3 节）。

## 2. 架构与默认值

```
client → prefix_router:8021 ──┬─→ rank0 API :8022 ─ DP0/TP8 ─┐
                             └─→ rank1 API :8023 ─ DP1/TP8 ─┤
                                                             ├─ Mooncake segment（宿主内存，pin）
                       mooncake_master :50088 ←──────────────┘
```

| 变量 | 默认 | 说明 |
|---|---|---|
| `DSV41_KV_POOL` | `1` | 0 = 完全回滚到旧行为 |
| `DSV41_MOONCAKE_SEGMENT_SIZE` | `32GB` | **每卡/die** 注册的宿主内存；须为 1GiB 整数倍 |
| `DSV41_KVPOOL_MIN_FREE_GIB` | `256` | 预检要求启动后至少保留的宿主内存 |
| `DSV41_KVPOOL_MAX_MEM_PCT` | `40` | 池总量不得超过 MemTotal 的该百分比 |
| `DSV41_MOONCAKE_PREFERRED_SEGMENT` | `1` | 每个 rank 写入**自己的** segment（A3 必需，见 3.1） |
| `DSV41_MOONCAKE_TRANSFER_TIMEOUT_MS` | `10000` | HIXL 连接/传输超时（替代默认 60 s） |
| `DSV41_MOONCAKE_MASTER` | `127.0.0.1:50088` | master RPC |
| `DSV41_MOONCAKE_METRICS_PORT` | `9003` | master metrics |
| `DSV41_MOONCAKE_BACKEND` | `mooncake` | 后端（mooncake / memcache / yuanrong） |
| `DSV41_MOONCAKE_TENANT` | `default` | 多租户隔离 |
| `DSV41_MOONCAKE_SSD_OFFLOAD` | `0` | 1 = 持久层（需 `*_SSD_PATH`） |
| `DSV41_KVPOOL_STOP_WITH_SERVE` | `0` | 1 = `STOP=1` 同时停 master |
| `DSV41_PYTHONHASHSEED` | `0` | 缓存 key 跨 rank 一致性 |

引擎侧参数由 `serve.sh` 自动追加：

```json
--kv-transfer-config '{"kv_connector":"AscendStoreConnector","kv_role":"kv_both",
  "kv_connector_extra_config":{"backend":"mooncake","lookup_rpc_port":"0"}}'
```

配置写入 `state/kvpool/mooncake.json`（`MOONCAKE_CONFIG_PATH`）。
`serve.sh` 启动前调用 `scripts/kvpool.sh ensure`：**先做容量预检**，再生成
配置 + 按需拉起 `mooncake_master`（端口已被监听则复用外部 master）。Docker
运行时下 master 在容器内启动；容器共享宿主 network/IPC/PID 命名空间，端点一致。

## 3. 内存预算（务必先算）

- 注册的内存是 **pin 住的宿主内存**，引擎退出时释放；
- 总量 = `SEGMENT_SIZE × 卡数`。本机 DP2×TP8 = 16 die：
  - **32GB/卡 → 512 GiB（默认）**
  - 16GB/卡 → 256 GiB
  - 64GB/卡 → 1 TiB（**本机不可用**）
- 实测边界（2026-11-05）：单进程注册 64 GiB 成功，16 并发注册 64 GiB 也成功
  （1 TiB 总量、宿主空闲时）；但**引擎启动时**还要占用数百 GiB 宿主内存，
  64GB/卡 × 16 会在 worker 侧失败：

  ```
  HcommMemReg failed, ret:19, size:68719476736 bytes
  register_local_memory_failed ... error=-1
  RuntimeError: Initialize mooncake failed.   (ret=-600)
  ```

  失败发生在模型加载完之后，代价是一整轮启动时间，还残留一批持锁 worker，
  需要 `STOP=1 bash scripts/serve.sh` 清理。
- 因此 `kvpool.sh` 在启动前做预检（`check-capacity`）：
  `SEGMENT_SIZE × TP × DP` 必须同时满足
  `≤ MemTotal × 40%` 且 `≤ MemAvailable − 256 GiB`，否则直接拒绝启动并给出
  建议值（4 GiB 对齐）。本机默认值：512 GiB ≤ 804 GiB 且 ≤ 1715 GiB，通过。
- 共享节点上要给其它租户留足余量；当前机器总内存 2 TiB，`/dev/shm` 已被
  其它任务占用约 220 GiB；内存吃紧时可下调 `SEGMENT_SIZE`（不影响开关语义）。

## 3.1 A3 上的跨卡 P2P：为什么必须 `preferred_segment=1`

默认情况下 master 会把新 block 分配到任意有空闲的 segment（`prefer_alloc_in_same_node`
只保证同机，不保证同 rank）。于是 rank5 写 block 时可能被分到 rank7 的宿主 segment，
这次拷贝走 HIXL 跨卡 P2P，在本机 A3 上失败：

```
E transfer_task.cpp:396] Failed to complete transfers after 60000 milliseconds
E transfer_executor_base.cpp:258] Connect timeout to: 172.16.0.114:20729
  halDeviceDisableP2P failed, drvRetCode=17, devIdDes=5, phyIdSrc=7
ERROR mooncake_backend.py:448] Failed to put 4 keys out of 4. error_codes=[-1]
```

后果不只是丢缓存：60 s 超时重试会持续堆积，最终 `Worker proc VllmWorker-5 died
unexpectedly`，整个引擎退出（watchdog 随后重启，形成循环）。

修复：`preferred_segment=true` 让每个 rank 把 block 写进**自己的** segment，
写路径退化为本地 NPU→宿主内存拷贝，不需要 P2P。跨 rank 读取仍走 HIXL，
但前缀路由会把同一前缀固定到同一 rank，所以正常流量几乎不触发；真触发时
`ASCEND_TRANSFER_TIMEOUT=10s` 保证快速失败而不是卡 60 s。

> 同机 16 卡共享一个 master 的前提是"各写各的 segment"。若将来驱动修好跨卡
> HIXL，可把 `DSV41_MOONCAKE_PREFERRED_SEGMENT=0` 打开以获得真正的跨 rank
> 负载均衡。

**实测结论（2026-11-05）**：只做 `preferred_segment=1` 还不够。写路径确实是
本地的（`Failed to put` 归零），但**读**路径仍会命中别的 rank 的 block：

```
E client_service.cpp:1139] Transfer failed for key: ...@group:11@cache_role:kv...
E transfer_executor_base.cpp:597] Transfer failed to: 172.16.0.114:20671
→ Worker_DP1_TP7 死亡 → EngineDeadError
```

而前缀路由并不能保证"同一 system prompt 只落一个 rank"：两个不同问题共享
system prompt 时，router 会按负载分到不同 rank，于是必然触发跨 rank 读。
所以在本机驱动支持跨卡 HIXL 之前，外部池不可用。

更糟的是崩溃会污染 HIXL：之后即使单独 `TransferEngine().initialize()` 也会报
`RunAicpuThreadInit ... aicpu exception` 并返回 -1。恢复方式（按代价递增）：

1. 重启服务（不带池）——vLLM 本身不需要 HIXL，可正常服务；
2. 若仍要试池：`npu-smi set -t reset -i <npu_id> -c <chip_id>` 复位对应芯片
   （会中断该芯片上的任务）；
3. 仍不行则重启节点。

启用前请先在**空闲芯片**上验证 HIXL 可用：

```bash
python3 - <<'PY'
import torch, torch_npu
from mooncake.engine import TransferEngine
torch.npu.set_device(0)
print(TransferEngine().initialize("127.0.0.1", "P2PHANDSHAKE", "ascend", ""))  # 0 = OK
PY
```

## 4. 启用 / 验证 / 回滚

### 启用（默认已启用）

```bash
# 容量预检（不启动任何进程）
bash scripts/kvpool.sh check-capacity

# 只启动 master（幂等；不影响正在运行的引擎）
bash scripts/kvpool.sh start
bash scripts/kvpool.sh status

# 完整启动（含预检 + master + 引擎）
bash scripts/serve.sh
```

启动日志应出现：

```
[kvpool] 容量校验通过：32GB/卡 × 16 = 512 GiB（...）
[serve] kv_pool=on master=127.0.0.1:50088 segment=32GB/card backend=mooncake config=.../state/kvpool/mooncake.json
[kvpool] ✅ kv pool 就绪 ...
```

引擎启动后确认外部命中：

```bash
curl -s http://127.0.0.1:8022/metrics | grep -E "external_prefix_cache_(hits|queries)_total"
# hits/queries 持续上升 = 外部池在生效
```

### 回滚（不改代码）

```bash
# config/dsv41.local.env
DSV41_KV_POOL=0
# 然后重启服务（冷启动）
STOP=1 bash scripts/serve.sh && bash scripts/serve.sh
```

回滚后引擎参数与之前完全一致（不再追加 `--kv-transfer-config`）。

### 停止 master

```bash
bash scripts/kvpool.sh stop        # 只停本脚本启动的 master
```

## 5. 注意事项

1. **DeepSeekV4 是混合注意力模型**：`kv_load_failure_policy=recompute`
   不支持（rc2 官方文档明确列出），保持默认失败策略；
2. **内存池不持久**：引擎退出 → 注册段释放 → 缓存清空。需要跨重启保留时
   启用 SSD offload（`DSV41_MOONCAKE_SSD_OFFLOAD=1` +
   `DSV41_MOONCAKE_SSD_PATH=/nvme/...`，每个 rank 独立子目录由连接器创建）；
3. **首次命中需要热身**：HCCS 单边连接惰性建立，参考实现建议先跑
   2–3× 设备数的 8k/1 请求；
4. **不要与其它实例共享 `state/`**：每实例独立 `DSV41_STATE_DIR`；共享同一
   master 时保持相同模型、缓存格式与 `PYTHONHASHSEED`；
5. Docker 运行时：`state/` 与项目目录按相同路径挂载，配置对容器可见；
   宿主未安装 `hccn.conf` 时连接器使用驱动默认配置（本项目已验证
   `hccn_tool` 可读）。

## 6. 与 DP lockstep 的关系

外部 KV 池最大的运行期收益是**减少重复 prefill**：DP2 lockstep 下，任何一个
rank 的长 prefill 都会让另一个 rank 的 decode 停顿（见
`docs/troubleshooting.md` 的 lockstep 一节）。长文档第二次问答直接命中外部
KV 后几乎没有 prefill 步，这类停顿随之消失——这是本方案对交互延迟的主要贡献，
而不是提升纯 decode 速度。

## 7. 相关命令

```bash
make kvpool-status       # master 状态 + 生效配置
make kvpool-start        # 仅启动 master
make kvpool-stop         # 停止 master
bash scripts/kvpool.sh check-capacity # 只做内存预算预检
bash scripts/kvpool.sh print-env   # 打印全部生效变量
bash scripts/kvpool.sh render-config  # 重新生成 mooncake.json
```
