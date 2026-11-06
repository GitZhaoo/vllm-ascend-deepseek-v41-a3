# 外部 KV 池（AscendStoreConnector + MemCache）

本文档描述本项目的外部 KV 池。实现对齐参考部署
`CJYLZS/vllm-ascend-deepseek-v41-a2` 的 `kv_offload.sh`：

- 连接器：`AscendStoreConnector`，`backend=memcache`；
- 池服务：`memcache_hybrid` 的 **MetaService**（纯 CPU 进程，不占 HBM/卡）；
- 传输：`host_shm`（宿主机共享内存）；
- 每个 rank 注册 `DSV41_KV_DRAM_GB` 宿主内存；
- 启用前强制检查 **0101**（MRV2 抢占恢复）正确性补丁。

> 本机 `config/dsv41.local.env` 默认 `DSV41_KV_POOL=0`。这是有意的
> opt-in 设计：先确认池能正常注册、业务确有长前缀复用收益，再打开。

## 1. 为什么是 MemCache/host_shm

参考部署在 A2/910B3 上已经验证过这条路径。两个硬约束必须保留：

1. **`KV_PROTOCOL=host_shm`**。`host_rdma` 依赖的
   `libmf_hybm_accoffload.so` 在本镜像里是 0 字节，加载即失败；
   `device_rdma` 单 rank 能过，但 8 rank 会 segfault。
2. **不要用 HIXL 做池内传输**。MetaService + host_shm 不经过跨卡 HIXL，
   因此绕开了本机 A3 910C 的 `halDeviceDisableP2P ret:17` 问题。
   容器运行时也不要额外注入 `-e LD_LIBRARY_PATH`，那会覆盖镜像里的
   CANN 路径。

## 2. 架构

```
client → prefix_router:8021 ──┬─→ rank0 API :8022 ─ DP0/TP8 ─┐
                             └─→ rank1 API :8023 ─ DP1/TP8 ─┤
                                                             ├─ host_shm 注册的 DRAM
                  MetaService :5100/:6100/:8100 ←────────────┘  (2 GB/rank)
                  (memcache_hybrid，CPU-only)
```

`serve.sh` 在启动 vLLM 前调用：

```bash
bash scripts/kvpool.sh ensure
```

它做三件事：

1. 校验内存预算与 0101；
2. 写 `state/kvpool/mmc-local.conf` 和 `mmc-meta.conf`；
3. 启动或复用 MetaService。

然后引擎获得：

```bash
MMC_LOCAL_CONFIG_PATH=state/kvpool/mmc-local.conf
--kv-transfer-config '{"kv_connector":"AscendStoreConnector","kv_role":"kv_both",
  "kv_connector_extra_config":{"backend":"memcache"}}'
```

## 3. 默认值与配置

| 变量 | 默认 | 说明 |
|---|---|---|
| `DSV41_KV_POOL` | `0` | 1 = 启用外部池；0 = 完全回滚到旧行为 |
| `DSV41_KV_DRAM_GB` | `2` | **每个 rank** 注册的宿主内存（16 die → 32 GiB） |
| `DSV41_KV_PROTOCOL` | `host_shm` | 本镜像唯一验证可用的传输 |
| `DSV41_KV_WORLD_SIZE` | `256` | memcache_hybrid local service 的世界大小 |
| `DSV41_KV_META_PORT` | `5100` | MetaService RPC |
| `DSV41_KV_STORE_PORT` | `6100` | config-store RPC |
| `DSV41_KV_METRICS_PORT` | `8100` | metrics；不能与 vLLM `--port` 相同 |
| `DSV41_KVPOOL_MIN_FREE_GIB` | `64` | 预检要求保留的宿主内存 |
| `DSV41_KVPOOL_MAX_MEM_PCT` | `25` | 池总量不得超过 MemTotal 的百分比 |
| `DSV41_KV_ALLOW_UNFIXED_TREE` | `0` | 1 = 明知源码缺 0101 仍启动（不推荐） |
| `DSV41_KVPOOL_STOP_WITH_SERVE` | `0` | 1 = `STOP=1` 同时停 MetaService |

`mmc-local.conf`：

```ini
ock.mmc.meta_service_url = tcp://127.0.0.1:5100
ock.mmc.local_service.config_store_url = tcp://127.0.0.1:6100
ock.mmc.log_level = info
ock.mmc.local_service.world_size = 256
ock.mmc.local_service.protocol = host_shm
ock.mmc.local_service.dram.size = 2GB
```

`mmc-meta.conf`：

```ini
ock.mmc.meta_service_url = tcp://127.0.0.1:5100
ock.mmc.meta_service.config_store_url = tcp://127.0.0.1:6100
ock.mmc.meta_service.metrics_url = http://127.0.0.1:8100
ock.mmc.log_level = info
```

## 4. 内存预算

- 注册的是宿主 DRAM，引擎退出时释放；
- 总量 = `DSV41_KV_DRAM_GB × DP × TP`；
- 默认 `2 × 2 × 8 = 32 GiB`，对 2 TiB 节点非常保守；
- 上调前先跑：

```bash
bash scripts/kvpool.sh check-capacity
```

预检要求池同时满足：

```text
总量 ≤ MemTotal × DSV41_KVPOOL_MAX_MEM_PCT / 100
总量 ≤ MemAvailable − DSV41_KVPOOL_MIN_FREE_GIB
```

超限时脚本会拒绝启动并给出建议的 `DSV41_KV_DRAM_GB`，不会等到模型加载
完才在 worker 里失败。

## 5. 0101 是正确性闸门

`0101-kvpool-mrv2-resume.patch` 清掉 MRV2 抢占恢复后遗留的
`_preempted_req_ids` 标记。缺它时，下一次 RUNNING delta 会用新的 block
表覆盖旧的，把**错误的 block** 关联到池 key；之后命中池就会读到别人的
KV，输出错误但日志不报错。

因此 `kvpool.sh` 在启动 MetaService 前检查源码树：

```bash
grep -q '_preempted_req_ids.discard(request.req_id)' \
  <tree>/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py
```

缺失时直接 `exit 8`，除非显式设置
`DSV41_KV_ALLOW_UNFIXED_TREE=1`。补丁由
`bash scripts/apply_patches.sh apply` 安装，`verify` 也会检查。

## 6. 启用 / 验证 / 回滚

### 启用

```bash
# 1) 预算 + 配置 + MetaService（不启动 vLLM）
bash scripts/kvpool.sh ensure
bash scripts/kvpool.sh status

# 2) 确认 0101 在树里
bash scripts/apply_patches.sh verify

# 3) 重启服务，让引擎加载连接器
STOP=1 bash scripts/serve.sh
DSV41_KV_POOL=1 bash scripts/serve.sh
```

若要在本机长期启用，把 `config/dsv41.local.env` 改成
`DSV41_KV_POOL=1` 并冷启动服务。

### 启动日志

```text
[kvpool] 容量校验通过：2GB/rank × 16 = 32.0 GiB（...）
[kvpool] ✅ MetaService 就绪 (pid=...)
[serve] kv_pool=on backend=memcache protocol=host_shm dram=2GB/rank meta=127.0.0.1:5100
```

引擎 worker 日志应出现：

```text
Creating v1 connector with name: AscendStoreConnector
```

MetaService metrics：

```bash
curl -s http://127.0.0.1:8100/metrics | grep memcache_total_capacity_bytes
```

- 启动前/无 rank 注册时，`dram` 容量是 `0`；
- 16 个 rank 注册完成后，`dram` 总量应接近 `32 GiB`；
- 运行中可用 `bash scripts/kvpool.sh status` 查看。

外部池命中指标：

```bash
curl -s --noproxy '*' http://127.0.0.1:8022/metrics \
  | grep -E 'external_prefix_cache_(hits|queries)_total'
```

`hits/queries` 持续上升说明池在生效。刚启动时池为空，需要先跑几次相同
前缀热身。

### 回滚

```bash
# config/dsv41.local.env
DSV41_KV_POOL=0

STOP=1 bash scripts/serve.sh
bash scripts/serve.sh
```

回滚后不再追加 `--kv-transfer-config`，也不启动 MetaService。

### 停止 MetaService

```bash
bash scripts/kvpool.sh stop
```

只停止本脚本记录在 `DSV41_KV_META_PID_FILE` 里的进程；如果检测到外部
MetaService，会提示但不会误杀。

## 7. 运维注意

1. **池不持久**：MetaService 退出或引擎退出后注册段释放，缓存清空；
   当前实现不提供 SSD/持久层。
2. **不要共享 `state/`**：每个实例独立 `DSV41_STATE_DIR`，否则配置和
   PID 会互相覆盖。
3. **端口冲突**：`5100/6100/8100` 不能与 vLLM `--port` 或其它服务冲突；
   `render-config` 会提前检查。
4. **Docker 运行时**：`state/` 以相同路径挂载，配置对容器可见；
   `serve.sh` 在容器内执行 `kvpool.sh ensure`，MetaService 随容器生命周期。
   因此 `docker stop` 会一并停止池；需要跨容器重启保留池时，改用
   `DSV41_RUNTIME=native` 或手工在宿主启动 MetaService。
5. **内核参数**：`host_shm` 使用宿主共享内存；确认 `/dev/shm` 足够大，
   且没有其它任务占满。池注册的是 DRAM 段，不依赖 `tmpfs` 文件大小，
   但引擎自身仍会使用 `/dev/shm`（Engram DP shared memory）。
6. **不要设置 `ASCEND_ENABLE_USE_FABRIC_MEM=1`**：本机驱动会拒绝大块 UC
   预留；该参数与 MemCache 池无关。

## 8. 排障

| 症状 | 原因 | 处理 |
|---|---|---|
| `MetaService 60s 内未就绪` | 端口占用 / Python 依赖缺失 / `/dev/shm` 不足 | 看 `logs/memcache_meta.log`；`ss -ltnp` 查 5100/6100/8100 |
| `DSV41_KV_PROTOCOL` 警告 `host_rdma` | 误配了不可用传输 | 改回 `host_shm` |
| `KV_OFFLOAD 已开启，但源码树不含 0101` | 补丁未应用 | `bash scripts/apply_patches.sh apply`；不要用 `ALLOW_UNFIXED_TREE` 绕过 |
| `memcache_total_capacity_bytes ... dram 0` | 引擎还没启动/rank 还没注册 | 等引擎 ready 后再查；若一直 0，查 worker 日志的 connector 初始化 |
| 池启动后 OOM | `DSV41_KV_DRAM_GB` 太大 | `bash scripts/kvpool.sh check-capacity` 后会给出建议值 |
| 端口 8100 与其它服务冲突 | metrics 端口复用 | 同时改 `DSV41_KV_METRICS_PORT`（或另起实例） |

## 9. 相关命令

```bash
make kvpool-status       # MetaService 状态 + 生效配置
make kvpool-start        # 仅启动 MetaService（幂等）
make kvpool-stop         # 停止本脚本启动的 MetaService
make kvpool-check        # 只做内存预算预检
bash scripts/kvpool.sh print-env        # 打印全部生效变量
bash scripts/kvpool.sh render-config    # 重新生成两个 conf
```
