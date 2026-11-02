# Prefix / KV-aware router（external LB）

DP2 下两个 EngineCore 的 **KV cache 互相独立**。默认的 `--data-parallel-size 2`
由一个 ApiServer 按 inflight / waiting / KV 使用率做负载均衡，同一段前缀可能落到
不同 rank，prefix cache 只能各自命中，跨 rank 无法复用。

本仓库提供 `scripts/prefix_router.py`：把 DP server 拆成两个独立的 API 端口，
由路由器根据每个 rank 的 **KV cache block 事件**把相同前缀的请求粘到同一 rank。
这是可选的生产路由模式，也是当前 16384 批次生产配置使用的模式。

> 不启用时保持 `DSV41_LB_MODE=internal`，行为与早期单端口 DP 完全一致。

## 1. 拓扑与端口

```
                         ┌─────────────────────────────────────┐
 client :8021 ──────────▶│ prefix_router.py                    │
 (OpenAI/SSE)            │  /v1/*  /health  /metrics            │
                         └───────┬─────────────────┬───────────┘
                                 │ longest prefix  │
                    ┌────────────▼───┐      ┌──────▼─────────┐
                    │ APIServer DP0  │      │ APIServer DP1  │
                    │ :8022          │      │ :8023          │
                    └───────┬────────┘      └──────┬─────────┘
                            │ KV events            │ KV events
                       tcp://*:5557          tcp://*:5558
                            │                      │
                    ┌───────▼────────┐      ┌──────▼─────────┐
                    │ EngineCore DP0 │      │ EngineCore DP1 │
                    │ TP8 / EP       │      │ TP8 / EP       │
                    └────────────────┘      └────────────────┘

              DPSupervisor :8020 只提供聚合 /health
```

| 端口 | 进程 | 用途 |
|---:|---|---|
| `8021` | `prefix_router.py` | 客户端入口；SDK / 客户端都指向这里 |
| `8020` | `DP supervisor` | 子 rank 存活聚合，供运维探活 |
| `8022` | `APIServer DP0` | rank0 OpenAI 兼容 API |
| `8023` | `APIServer DP1` | rank1 OpenAI 兼容 API |
| `5557` / `5558` | EngineCore ZMQ PUB | 每个 rank 的 KV cache 事件 |

端口均可覆盖：

```bash
DSV41_PORT=8021                  # router
DSV41_DP_SUPERVISOR_PORT=8020
DSV41_DP_BASE_PORT=8022          # rank0；rankN = base + N
DSV41_KV_EVENTS_BASE_PORT=5557
```

> 单机 2 个 DP rank 时 `--data-parallel-size-local=2`；多机部署需要按节点拆分
> local size 与 start rank，本仓库当前只验证单机 A3。

## 2. 启用

```bash
# config/dsv41.local.env
DSV41_LB_MODE=external
DSV41_DP=2
DSV41_TP=8
DSV41_KV_EVENTS=1
DSV41_DP_SUPERVISOR_PORT=8020
DSV41_DP_BASE_PORT=8022
DSV41_KV_EVENTS_BASE_PORT=5557
DSV41_ROUTER_MAX_SKEW=8

bash scripts/serve.sh
STATUS=1 bash scripts/serve.sh
```

`serve.sh` 会：

1. 用 `--data-parallel-multi-port-external-lb` 启动 DP supervisor；
2. 子进程自动得到 `--data-parallel-rank 0/1`、端口 `8022/8023`；
3. 给 vLLM 传 `--kv-events-config`；publisher 会按 rank 自动把端口偏移到
   `5557/5558`；
4. 启动 router，监听 `DSV41_PORT`，把请求转发到两个 rank；
5. 健康检查仍以 `http://127.0.0.1:8021/health` 为准。router 在**两个 rank 都
   健康**时才返回 200；任一 rank 未就绪时返回 503。

只重启路由器（不动 vLLM，约 1 秒中断）：

```bash
ROUTER_RESTART=1 bash scripts/serve.sh
```

## 3. 路由算法

### 3.1 KV 事件 → 前缀指纹链

vLLM 的 `BlockStored` 事件包含：

- `block_hashes`：本次新写入的 block hash；
- `parent_block_hash`：前缀链上的父 block；
- `token_ids`：这些 block 的 token；
- `block_size`：默认 128。

router 不依赖 vLLM 内部的 hash 算法，而是自己维护一条 **token 指纹链**：

```
digest_0 = BLAKE2b(tokens[0:128])
digest_i = BLAKE2b(digest_{i-1} || tokens[i*128:(i+1)*128])
```

每个 rank 保存 `set[digest]`。收到请求时：

1. 调一次 rank0 的 `/tokenize` 得到 prompt token（按 messages/prompt 缓存，
   避免重复 tokenize）；
2. 计算 prompt 的前缀 digest 链；
3. 对每个 rank 求“从第一个 block 开始连续命中的最长长度”；
4. 选命中最长者；并列时选当前 active 请求更少的 rank。

同一 block 会被 12 个 KV cache group 各发一次事件；router 用引用计数去重，
`BlockRemoved` 逐次递减，避免某个 group 移除就让整段前缀“消失”。

### 3.2 没有命中时

- 先查 `sticky` 表（`prompt body hash → rank`）；命中则继续用该 rank；
- 否则选 active 最少的 rank，并记录 sticky；
- sticky 表最多 50k 条，LRU 淘汰；
- 如果目标 rank 的 active 比另一个多出 `DSV41_ROUTER_MAX_SKEW`（默认 8），
  强制切到另一个 rank，防止单一热前缀把 16 张卡中的一半长期闲置。

> 这是一个**前缀亲和（prefix-affinity）**路由器，不是全局最优调度器。它优化
> 的是“相同 system prompt / 长文档重复出现”的场景；对完全随机、短 prompt 的
> 负载，收益接近 0，额外成本是每请求一次本地 `/tokenize`（通常几毫秒）。

### 3.3 KV 事件的边界情况

- **短于 128 token 的 prompt 不会产生 `BlockStored`**，因此没有亲和信息；
  这类请求走 sticky / 最少 active 逻辑即可。
- 事件流从 router 启动时开始消费。router 重启后，旧 KV cache 不会重放，
  需要新请求重新填充 digest；如需干净状态可重启 vLLM 或调用 vLLM 的
  reset prefix cache 接口。
- `AllBlocksCleared` 会清空该 rank 的 digest 表。
- 事件积压时 publisher 有 HWM；router 只保留最近 200k 个 digest，超出后整体
  重建（宁可少命中，不无限吃内存）。

## 4. 观测

```bash
# 健康（两个 rank 都健康才 200）
curl --noproxy '*' http://127.0.0.1:8021/health

# 路由统计；配置了 API key 后必须带 Bearer
curl --noproxy '*' http://127.0.0.1:8021/router/stats \
  -H "Authorization: Bearer $DSV41_API_KEY"
```

返回示例：

```json
{
  "ranks": [
    {"url": "http://127.0.0.1:8022", "active": 0, "healthy": true, "cached_prefix_digests": 97},
    {"url": "http://127.0.0.1:8023", "active": 0, "healthy": true, "cached_prefix_digests": 69}
  ],
  "requests": 68,
  "affinity_hits": 61,
  "fallback": 17,
  "events": 4236
}
```

- `cached_prefix_digests`：该 rank 当前可匹配的前缀块数；
- `affinity_hits`：按最长前缀命中路由的请求数；
- `fallback`：无命中 / sticky / skew 切换的请求数；
- `events`：消费的 KV 事件条数。

`GET /metrics` 会把两个 rank 的 Prometheus 文本合并（去重 HELP/TYPE），
因此 `prefix_cache_hits_total`、`num_requests_running` 等仍可直接抓取。
单 rank 指标也可直接访问 `:8022/metrics`、`:8023/metrics`。

## 5. 故障与回滚

| 症状 | 原因 | 处理 |
|---|---|---|
| `/health` 503，但两个 rank `/health` 200 | router 尚未收到健康轮询结果 | 等 5 s；检查 `logs/router.log` |
| `/health` 503，某个 rank 不健康 | 子 rank 启动失败 / 崩溃 | 看 `logs/serve_*.log`；watchdog 会整栈重启 |
| `/router/stats` 401 | 已启用 stats token | 带 `Authorization: Bearer $DSV41_API_KEY` |
| 路由全部 fallback、`cached_prefix_digests=0` | 没有 KV 事件 / prompt < 128 token / router 刚重启 | 发一个 >128 token 的请求；检查 5557/5558 |
| 端口 5557/5558 看不到 | 本机 `ss` 在某些 netns 下漏报 ZMQ | `grep -E ':15B5\\|:15B6' /proc/net/tcp`（5557/5558 十六进制） |
| 两个 rank 负载长期倾斜 | 热前缀过大 / skew 设置过小 | 调大 `DSV41_ROUTER_MAX_SKEW`，或拆分 system prompt |
| 想完全回退 | 不需要前缀亲和 | `DSV41_LB_MODE=internal` + `STOP=1 bash scripts/serve.sh` 后重启 |

router 只做转发，不修改返回体；上游 401/404/429 等状态码和 SSE 流会原样透传。
API key 仍由两个 rank 的 `--api-key` 校验，router 会把客户端的 `Authorization`
头转发给 `/tokenize` 与推理接口。

## 6. 安全说明

- `/router/stats` 默认要求 `DSV41_API_KEY`（也可用 `DSV41_ROUTER_STATS_TOKEN`
  单独指定）；未设置 key 时才匿名开放。
- `/metrics` 与传统 vLLM 一样默认无鉴权；如需限制，请在本地网络层控制
  `/metrics`、`/router/stats` 的访问。
- router 进程命令行**不再包含 API key**；key 通过环境变量继承。
