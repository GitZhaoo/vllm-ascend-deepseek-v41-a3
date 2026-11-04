# 外部 KV 池（Mooncake / AscendStoreConnector）

本文档描述本项目**默认启用**的外部 KV 池：把 prefix cache 的 KV block 从
NPU HBM 扩展到**宿主内存**（Mooncake Store），让重复前缀（长文档、多轮对话、
共享 system prompt）在跨请求、跨 DP rank 之间命中，而不是重复 prefill。

参考实现：ModelScope `x0ken0/vllm-ascend-a2-dsv41f`（A2 / 910B3 / Docker）。
本项目在其基础上按 A3（8×910C、DP2×TP8、native rootfs）适配。

## 1. 作用与收益

| 维度 | 无外部 KV 池 | 有外部 KV 池（默认 64GB/卡） |
|---|---|---|
| KV 存放 | 仅 NPU HBM：1,268,538 tokens/rank（≈6.7 GiB） | HBM + 宿主内存：64 GiB/卡 × 16 卡 = **1 TiB** |
| 等效容量 | 1.27M tokens/rank | ≈ **11.6M tokens/卡（≈9×）** |
| 复用范围 | 单 rank 生命周期内 | 跨请求、跨 DP rank；同一 master 下可跨实例 |
| 长文档二次提问 | 重新 prefill（慢，且 DP lockstep 阻塞另一 rank） | 直接命中，几乎免 prefill |
| 重启后 | 空 | 空（除非启用 SSD offload） |

实测参考：KV 单价约 **5.5 KB/token**（6.7 GiB / 1,268,538）。
容量换算：`tokens ≈ 64 GiB / 5.5 KB ≈ 11.6M`。

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
| `DSV41_MOONCAKE_SEGMENT_SIZE` | `64GB` | **每卡/die** 注册的宿主内存；须为 1GiB 整数倍 |
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
`serve.sh` 启动前调用 `scripts/kvpool.sh ensure`：生成配置 + 按需拉起
`mooncake_master`（端口已被监听则复用外部 master）。Docker 运行时下 master
在容器内启动；容器共享宿主 network/IPC/PID 命名空间，端点一致。

## 3. 内存预算（务必先算）

- 注册的内存是 **pin 住的宿主内存**，引擎退出时释放；
- 总量 = `SEGMENT_SIZE × 卡数`。本机 DP2×TP8 = 16 die：
  - 64GB/卡 → **1 TiB**
  - 32GB/卡 → 512 GiB
  - 16GB/卡 → 256 GiB
- 共享节点上要给其它租户留足余量；当前机器总内存 2 TiB，可用约 1.6 TiB，
  `/dev/shm` 已被其它任务占用约 220 GiB；
- 注册失败（内存不足 / 配额）时引擎会启动失败，日志关键字：
  `Initialize mooncake failed`、`Memory_Allocation_Failure(EL0004)`。

## 4. 启用 / 验证 / 回滚

### 启用（默认已启用）

```bash
# 只启动 master（幂等；不影响正在运行的引擎）
bash scripts/kvpool.sh start
bash scripts/kvpool.sh status

# 完整启动（含 master + 引擎）
bash scripts/serve.sh
```

启动日志应出现：

```
[serve] kv_pool=on master=127.0.0.1:50088 segment=64GB/card backend=mooncake config=.../state/kvpool/mooncake.json
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
bash scripts/kvpool.sh print-env   # 打印全部生效变量
bash scripts/kvpool.sh render-config  # 重新生成 mooncake.json
```
