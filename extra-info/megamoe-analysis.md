# MegaMoe（`enable_fused_mc2=2`）机制分析与调参矩阵

分析对象：DeepSeek-V4.1-Flash W8A8 / 单机 8×910C(A3) / TP8×DP2（EP16）/ DSpark K=5 /
BAT=16384 / max_num_seqs=32 / 图桶 ≤192 / `FULL_DECODE_ONLY` / 0202 limit_core_num 默认开。
源码基线：`new_rl:/cache/dsv41-rc2-rootfs/vllm-workspace/vllm-ascend/vllm_ascend`
（含 Lead 的 prefill-only 门槛补丁；`ascend_forward_context.py.orig0203` 为其打补丁前副本）。
实验数据：`new_rl:/cache/exp/json/*.json`、`/cache/exp/logs/*`（只读）。

--------------------------------------------------------------------------------

## 0. TL;DR（先看这一节）

1. **MegaMoe 在这台机器/这个模型上从来没有真正跑过。** `enable_fused_mc2=2`
   会被 `AscendConfig._is_megamoe_supported_by_config()` **静默回滚成 0**，
   因为 V4.1-Flash 的 `moe_intermediate_size=2304` 不满足 A3 白名单的
   `moe_intermediate_size % 512 == 0`（2304 = 4.5×512）。
   exp29 / exp41 / exp42 的引擎日志里都有
   `MegaMoe is not supported for this model config; additional_config.enable_fused_mc2 will be set to 0.`
   （每 rank 重复约 20 次），且**没有任何** `CANN MegaMoe sym-buffer alloc` 日志。
2. 于是 exp41/exp42 相对 exp40 的真实差异只剩一个：
   `multistream_overlap_shared_expert` 被 `enable_fused_mc2` 的互斥逻辑
   **顺带强制关掉了**（`ascend_config.py:842`，先于回滚执行）。
   即当前 A/B 实际上是 **"shared-expert 多流 overlap 开 vs 关"**，不是 MegaMoe。
3. 实测与此完全一致：TTFT 没有真实收益（cold TTFT：基线 276.8/281.4ms、
   exp29 280.1、exp41 278.6、exp42 278.9ms），单流 decode 稳定掉 ~5%
   （cold decode 109.2/109.8 tol/s vs 基线 113.9/114.1；dual c=2 stream 94.8–97.9
   vs 100–103）。exp42（prefill-only 门槛）与 exp41 **数值等价**，因为门槛在
   mega 不可达时是死代码。这正好是丢掉 shared-expert overlap 的代价，
   而不是 MegaMoe 的代价。
4. **建议：生产保持 `enable_fused_mc2=0`（默认），不要带 `=2`。**
   exp41/exp42 可以当作"关掉 multistream overlap"的对照结论使用。
   要继续验证 MegaMoe 必须先绕过形状白名单（`0311` 补丁，env 默认关），
   且即使绕过成功，`enable_fused_mc2=1` 依旧会强制关 multistream，
   prefill-only 门槛本身**不能**把 decode 的 −5% 还回来。
5. 若将来真的启用（假设 0311 能让算子接受该形状）：
   **门槛推荐 `VLLM_ASCEND_V41_MEGAMOE_PREFILL_MIN_TOKENS=512`**（256 为激进备选），
   **draft 建议排除**（`0310` 补丁，env 默认关）；
   128 会污染 decode 高桶，1024/2048 会把中尺寸 prefill chunk 降级到 MC2，均不推荐。

--------------------------------------------------------------------------------

## 1. 实测复核：exp29/exp41/exp42 里 MegaMoe 是否真的启用？

### 1.1 证据链（全部可在远端只读日志复核）

| # | 证据 | exp29（08:54 mega-all） | exp41（14:20 mega-all） | exp42（14:30 prefill-only） |
|---|------|--------------------------|--------------------------|------------------------------|
| 1 | `additional_config.enable_fused_mc2` 入参 | 2 | 2 | 2 |
| 2 | 引擎日志 `MegaMoe is not supported ... set to 0` 次数 | 20 | 20 | 20 |
| 3 | `CANN MegaMoe sym-buffer alloc` 次数 | 0 | 0 | 0 |
| 4 | `Shared expert split computation validation passed`（overlap 生效标志） | 0 | 0 | 0 |
| 5 | weights / peak activation 内存（worker.py:933） | 38.74 / 8.18 GiB | 38.74 / 8.18 GiB | 同 |
| 6 | ACL graph capture 耗时 | 73–74s | 73–74s | 同 |
| 7 | NPU graph memory | 0.64–0.65 GiB | 0.64–0.65 GiB | 同 |

对照基线 exp40（`enable_fused_mc2` 缺省 = 0）：
`Shared expert split computation validation passed` 出现 2 次、没有回滚告警、
graph capture 75s、graph memory 0.68 GiB。

日志位置：
- exp29 → `/cache/exp/logs/serve_dsv41w8a8rc2-20261112_085433.log`（第 98–99 行起）
- exp40 → `/cache/exp/logs/serve_dsv41w8a8rc2-20261112_141056.log`
- exp41 → `/cache/exp/logs/serve_dsv41w8a8rc2-20261112_142053.log`（第 100–101 行起）
- exp42 → `/cache/exp/logs/serve_dsv41w8a8rc2-20261112_143045.log`

第 4 条特别关键：`AscendSharedExperts.__init__` 只有在
`ascend_config.multistream_overlap_shared_expert == True` 时才会包一层
`process_weights_after_loading` 校验并打印这条 INFO。exp29/41/42 一条都没有，
说明进程内该开关已经是 False。

### 1.2 为什么会被回滚（源码链路）

`vllm_ascend/ascend_config.py`

```python
# AscendConfig._validate_user_input_ranges()（model_validator(mode="after")）
if self.enable_fused_mc2 in (0, 1):
    _MEGA_MOE_SUPPORTED = False                       # 默认/rollback = dispatch_ffn_combine
elif self.enable_fused_mc2 == 2:
    _MEGA_MOE_SUPPORTED = importlib.util.find_spec("cann_ops_transformer") is not None
    self.enable_fused_mc2 = 1                         # 注意：先降成 1（= 后续都按 fused 处理）
```

随后 `derive_and_validate()`（同一 config 初始化流程内）：

```python
# ascend_config.py:842
if self.enable_fused_mc2 == 1 and self.multistream_overlap_shared_expert:
    self.multistream_overlap_shared_expert = False    # ← 副作用先发生
    logger.warning_once("enable_fused_mc2 and multistream_overlap_shared_expert cannot be enabled ...")
# ascend_config.py:848
if self.enable_fused_mc2 == 1 and is_mega_moe_supported() and not self._is_megamoe_supported_by_config(vc):
    self.enable_fused_mc2 = 0                         # ← 静默回滚，但 STOP 不回滚 multistream
    logger.warning_once("MegaMoe is not supported for this model config; ...")
```

`_is_megamoe_supported_by_config()`（`ascend_config.py:1052`）的 A3 分支：

```python
hidden_size = 5120                      # 5120 in [1024, 8192] 且 %512==0 → 通过
moe_intermediate_size = 2304            # 2304 % 512 == 256 → 直接 return False（无 warning！）
if moe_intermediate_size < 1024 or moe_intermediate_size > 3072 or moe_intermediate_size % 512 != 0:
    return False
```

算子包本身是存在的：`/cache/dsv41-rc2-rootfs/usr/local/Ascend/cann-9.2.0/python/site-packages/cann_ops_transformer`
（`serve.sh` 把 `$CANN/python/site-packages` 加进 PYTHONPATH），所以
`find_spec("cann_ops_transformer")` 为 True——**"包存在"不等于"形状被允许"**，
这正是本次误判的来源。

附：A5（950）分支 `_is_a5_megamoe_supported_by_config()` 对
`intermediate_hidden = 2*moe_intermediate_size = 4608` 也不在白名单
`{1024,2048,3072,4096,7168,...}` 里——即该模型形状在 A3/A5 两套检查下都会被拒，
不是单一硬件的白名单过窄。

### 1.3 实测数字对比（同一 bench 定义，来自 `/cache/exp/json`）

| 实验 | 生效配置 | single_cold TTFT | cold decode tok/s | single_hot TTFT | single_hot decode | dual c=2 aggregate/stream/TTFT | dual c=16 | dual c=64 |
|------|----------|------------------|-------------------|-----------------|-------------------|-------------------------------|-----------|-----------|
| exp36 | baseline | 281.4 | 114.13 | 249.0 | 101.53 | 165.0 / 103.0 / 692 | 910.5 agg | 1743.9 agg |
| exp40 | baseline（最终） | 276.8 | 113.88 | 297.8 | 103.63 | 152.6 / 100.0 / 780 | 1010.8 agg | 1683.0 agg |
| exp29 | fused_mc2=2→**0**，multistream **off** | 280.1 | 103.23 | 242.1 | 97.47 | 148.5 / 94.8 / 750 | 950.0 agg | 1861.6 agg |
| exp41 | 同上 | 278.6 | 109.17 | 无效（全 1token） | — | 152.6 / 97.1 / 744 | 986.8 agg | 1882.7 agg |
| exp42 | 同上 + prefill-only 门槛（512） | 278.9 | 109.81 | 294.9 | 99.14 | 157.4 / 97.9 / 693 | 994.9 agg | 1821.7 agg |

**exp42 是决定性验证**：门槛补丁（`PREFILL_ONLY=1, MIN_TOKENS=512`）打开后，
cold TTFT 278.9ms、cold decode 109.81 tok/s、dual c=2 stream 97.9 —— 与 exp41
（278.6 / 109.17 / 97.1）**完全一致**，与 exp29（280.1 / 103.2 / 94.8）也一致；
因为 mega 根本没启用，门槛是死代码，两种臂在源码语义上等价。
这从实验上二次确认 §1.2 的回滚链路。

读法：
- **prefill/TTFT 无差异**：cold TTFT（同一 prompt、同口径）276.8 / 281.4 / 280.1 / 278.6
  四组完全在噪声内；exp40 的 single_hot 297.8 是高侧离群点（exp36 同配置只要 249.0），
  "242ms vs 265–299ms、TTFT −20~25%" 是拿离群 baseline 比出来的，不能成立。
- **低并发 decode 稳定 −5%**：cold decode 103.2/109.2 vs 113.9/114.1，
  dual c=2 stream 94.8/97.1 vs 100.0/103.0。两个独立 run（exp29/41）方向一致、
  幅度一致 ⇒ 是配置的系统性差异，不是噪声。
- 高并发 aggregate（c=64）反而 1862–1883 vs 1683–1744，属于高并发方差 +
  acceptance 波动，不能当作收益。

--------------------------------------------------------------------------------

## 2. MegaMoe 机制（按源码逐层拆）

### 2.1 选择路径：A3 走 `FUSED_OR_CAPACITY`

`get_current_hardware_profile().moe_comm_policy`：
- A3（910C）→ `MoECommPolicy.FUSED_OR_CAPACITY`（`device/hardware_profile.py:298-310`，
  且 `_A3_CAPABILITIES` 含 `CANN_MEGAMOE`）
- A5 → `CAPACITY_AND_WORLD_SIZE`；A2 → `CAPACITY_AND_EXPERT_DENSITY`

对应的选择器就是 Lead 定位的
`_select_fused_or_capacity_moe_comm_method()`（`ascend_forward_context.py`，
打补丁前在 ~L318，当前在 ~L335）：

```python
megamoe = use_cann_megamoe(vllm_config)
if megamoe:                                   # ← 只看 token 数以外的东西
    ...                                        # Lead 的 prefill-only 门槛加在这里
    return MoECommType.FUSED_MC2               # 原版：prefill/decode 一律 FUSED_MC2
elif get_ascend_config().enable_fused_mc2 == 1 and get_ep_group().world_size <= 32:
    return MoECommType.FUSED_MC2               # dispatch_ffn_combine 老路径
if num_tokens is None or num_tokens <= mc2_tokens_capacity:
    return MoECommType.MC2
return MoECommType.ALLTOALL
```

`use_cann_megamoe(vllm_config)`（`ascend_forward_context.py:~L87`）的条件：

```python
is_mega_moe_supported()                          # 全局：enable_fused_mc2==2 且包存在
and hardware.supports(CANN_MEGAMOE)              # A3 True
and get_ascend_config().enable_fused_mc2 == 1    # 回滚后此处为 0 ⇒ False
and is_moe_model(vllm_config)
and parallel_config.enable_expert_parallel
and 1 < get_ep_group().world_size <= 64          # EP16 ✓
and lora_config is None
```

注意两点：
- 该 selector **完全不看** `is_draft_model` / `draft_moe_quant_type`（A5 的
  `_select_capacity_and_world_size...` 才看）⇒ 一旦 mega 生效，draft 也会被选中 FUSED_MC2。
- `num_tokens` 传入的是 `max(num_tokens_across_dp)`（`set_ascend_forward_context`
  里 `max_num_tokens = num_tokens_across_dp.max().item()`），即 DP 两组里较大的那个。

### 2.2 `mc2_tokens_capacity` 的连带变化（容易忽略）

`set_mc2_tokens_capacity()`（`ascend_forward_context.py:~L252`）：

```python
if ascend_config.enable_prefill_mc2 or (use_mega_moe and not decode_only_node):
    max_num_tokens = scheduler_config.max_num_batched_tokens      # 16384
elif cudagraph_capture_sizes:
    max_num_tokens = compilation_config.max_cudagraph_capture_size # 192
else:
    max_num_tokens = max_num_reqs * uniform_decode_query_len
num_tokens_per_tp_rank = ceil(max_num_tokens / tp_size)           # 16384/8 = 2048
if ascend_config.enable_fused_mc2:
    if use_mega_moe: num_tokens_per_tp_rank = min(rank, 16384)     # _MEGA_MOE_TOKENS_PER_RANK_LIMIT
    else:            num_tokens_per_tp_rank = min(rank, 512)       # _DISPATCH_FFN_COMBINE...
else:
    num_tokens_per_tp_rank = min(rank, 512)
_mc2_tokens_capacity = num_tokens_per_tp_rank * tp_size
```

- 基线（无 mega）：capacity = max_cudagraph_capture_size = **192**。
- mega 生效（哪怕有 prefill-only 门槛）：capacity = **16384**（每 rank 2048）。

后果：
1. 门槛把某 step 踢回普通路径时，`num_tokens <= 16384` 恒成立 ⇒ 回落目标是 **MC2，
   永远不会是 ALLTOALL**。基线里任何 >192 的 prefill chunk 是走 ALLTOALL 的，
   所以"启用 mega + 门槛"会把中尺寸 chunk 的通信算法从 ALLTOALL 换成 MC2，
   这是一个**独立于 MegaMoe 本身的 A/B 变量**（exp42 如果真跑 mega，就会混入这个变量）。
2. `TokenDispatcherWithMC2` 用 capacity 派生 `max_num_tokens_per_rank` /
   `global_bs`（`token_dispatcher.py:207-224`）：2048 vs 24，进而决定 MegaMoe
   symm buffer 的尺寸（见下）。`global_bs = _max_global_bs if skip_allreduce else 0`，
   而 `should_skip_allreduce_across_dp_group()` 在 `is_mega_moe_supported()` 时
   一律 `return False`（`utils.py:1328`，"MegaMoe 不支持动态 bs"）⇒ global_bs=0。
   我们这套拓扑本来也是 False（非 KV consumer），所以 DP allreduce 行为无额外损失；
   但如果是 PD 分离的 decode 节点，mega 会禁用 skip-allreduce 优化，需另算账。
3. `_MEGA_MOE_TOKENS_PER_RANK_LIMIT=16384` 在当前 BAT 下不会触发（2048 << 16384）。

### 2.3 symm buffer：大小、来源、与 ACL graph 的关系

调用链（远端源码 `ops/fused_moe/moe_comm_method.py`）：

```python
# FusedMC2CommImpl.__init__（~L289）
if enable_fused_mc2 == 1 and is_mega_moe_supported():
    self.mega_moe_symm_buffer = None
    self.get_symm_buffer_for_mega_moe, self.mega_moe = moe_utils.load_cann_mega_moe_ops()
    self.mega_moe_activation_kwargs = moe_utils.select_mega_moe_activation_kwargs(...)

# 首次真正执行 mega 时才分配（_apply_cann_mega_moe → _init_mega_moe_symm_buffer）
if self.mega_moe_symm_buffer is None:
    self.mega_moe_symm_buffer = self._init_mega_moe_symm_buffer(
        dispatch_quant_mode, dispatch_quant_out_dtype, is_decode_only_node=...)
```

`_init_mega_moe_symm_buffer`（~L386-467）的尺寸推导（**必须所有 EP rank 一致**）：

```python
num_max_tokens_per_rank = token_dispatcher.global_bs // ep_world_size   # global_bs=0 时走下一支
                        = token_dispatcher.max_num_tokens_per_rank      # = capacity/tp = 2048
num_topk    = experts_per_token                                     # 6
num_experts = 384
expert_per_rank = num_experts // ep_world_size                      # 384/16 = 24
absolute_safe_max_recv_token_num = num_max_tokens_per_rank * ep_world_size * min(num_topk, expert_per_rank)
                                 = 2048 * 16 * 6 = 196608
max_recv_token_num = mega_moe_max_tokens (默认 65536)               # P/PD-mixed 节点
                     = min(65536, 196608) = 65536
get_symm_buffer_for_mega_moe(group, num_experts=384, num_max_tokens_per_rank=2048,
    num_topk=6, hidden=5120, intermediate_hidden=2*intermediate_size_per_partition(≈2304)=4608,
    max_recv_token_num=65536, dispatch_quant_mode=2(W8A8 INT8), dispatch_quant_out_dtype=258)
```

要点：
- **不是 torch tensor，也不是 `HCCL_BUFFSIZE`。** 它由 CANN 的
  `CommContextManager` 分配一块 **custom CCL buffer**（`cann_ops_transformer/ops/mc2/mega_moe/mega_moe.py:273+`，
  `customCclBufferSize = _get_mega_moe_ccl_buffer_size(...)`），
  与 `HCCL_BUFFSIZE` 环境变量无关（源码注释也明确写了 "symm buffer (separate torch alloc,
  not HCCL_BUFFSIZE)"）。因此调 `HCCL_BUFFSIZE` 不会缩小/放大 mega 的通信内存。
- `max_recv_token_num` 是 **dispatch 后每 rank 可能收到的最大 token 数**；超了会
  "precision degradation"（源码 warning 原文），所以默认 65536 是按 prefill
  极端路由倾斜留的量。decode-only 节点会自动缩到 `absolute_safe_max_recv_token_num`。
- 内存量级（估算，算子内部公式未导出）：dispatch 接收 int8 `65536×5120 ≈ 0.33GiB`、
  combine 输出 bf16 `65536×5120×2 ≈ 0.67GiB`、L1 中间 `65536×4608 int8 ≈ 0.30GiB`，
  再乘上镜像/scale/expert token 元数据 ⇒ **单 rank 约 1.5–2.5 GiB 量级**，
  384 experts target 与 128 experts draft 是两个独立的 `MoECommMethod` 实例
  （`_moe_config_key = (MoECommType, num_experts, num_local_experts)`），
  若 draft 也走 mega 会**再分配一份**。
- 注释强调尺寸只能来自 rank-invariant 配置（不能用当前 forward 的 per-rank token 数），
  否则各 rank 分配时机/大小不一致 ⇒ HCCL `SUSPECT REMOTE ERROR 507057`。
  这也是为什么它挂在 lazy 分配上仍然安全：参数全部来自 compile-time config。

**与 ACL graph capture 的关系：**
- 我们有 `cudagraph_mode=FULL_DECODE_ONLY`（`separate_routine()` ⇒ prefill eager、
  decode 按 ≤192 的桶捕获）。mega 算子**本身是可捕获的**：
  `/cache/.../cann_ops_transformer/ops/mc2/mega_moe/graph_convert_mega_moe.py`
  注册了 torchair GE converter（`MegaMoe(...)`，专门给 graph mode 用）。
- 但 symm buffer 的创建是一次 **EP 集合通信 + CANN context 创建**（
  `create_context()`），属于 host 侧集合操作，**必须在 graph capture 之前完成**。
  正常流程里 profile/warmup 的 eager dummy run 会先命中 mega 完成分配；
  如果某个 decode 桶是进程里第一次 mega 调用，就会在 capture 内部做集合握手，
  风险是 hang / abort / 捕获到一次性初始化——这是 prefill-only 门槛的一个
  *意外好处*：prefill 一定先于 decode capture 命中 mega（eager），分配时序天然安全。
- A2/A3 上 `x_active_mask` 必须非空（A5 必须为 None）：mask 来自
  `set_mc2_mask()` 预分配的 `max_num_batched_tokens(=16384)` bool 缓冲，
  每次 forward 切 `[:padded_num_tokens]` 再 `.to(torch.int8)`。decode 桶里
  padded 长度按桶固定，capture 后每张图固定；`global_bs==0` 时才会传 mask。

### 2.4 如果 MegaMoe 真的启用，为什么 decode 小 batch 会更慢

按影响从大到小：

1. **丢掉 `multistream_overlap_shared_expert`（互斥，配置级强制）——实测就是 −5%。**
   `AscendSharedExperts` 把 shared-expert 的 Gate-Up/Down 放到
   `shared_experts_calculation_stream()` 上，与 routed dispatch / GMM1 / combine
   按 event 里程碑重叠（`ops/fused_moe/shared_experts.py:460-630`）。
   decode 每 step 只有 ≤192 token，shared expert 的计算量占比反而高；
   overlap 关掉后 Gate-Up→激活→Down 全部串到主流，正好是 ~0.4–2ms/step 的量级。
   而且这个开关是 **config-init 时一次性决定**，decode 走 MC2 也一样是关的——
   所以 prefill-only 门槛救不回 decode。
2. **融合算子在极小 M 下效率低。** decode 每 rank 只有 192/8=24 个 token，
   dispatch 后每个 expert 分到几个 token；MegaMoe 把 dispatch+GMM1+激活+GMM2+combine
   收进一个算子，省掉的是 host launch / 中间搬运，但**算子内部仍是完整的
   dispatch/combine 握手 + 两次分组矩阵乘**，小 M 时 GMM 的 cube 利用率极低，
   融合收益抵不过固定开销。相比之下 MC2 的 dispatch/combine 是专门为小 batch
   调过的两段式（`_MC2_TOKENS_PER_RANK_LIMIT=512`）。
3. **固定尺寸 symm buffer / CCL context 的资源占用与调度不随 token 数缩小。**
   MC2 的 HCCL buffer 需求由 `HCCL_BUFFSIZE=1024MB` 决定、按 global_bs=0 的动态模式跑；
   mega 一旦分配就按 2048 tokens/rank + 65536 recv 的固定 buffer 占着通信资源，
   decode 时大量容量闲置。
4. **量化路径本身不是瓶颈，但每层有权重列表开销。**
   W8A8 对应 `_QUANT_SETTING_MAP[W8A8] = (INT8 mode=2, int8 out, int8 weight)`
   （`moe_utils.py:44-60`），权重在 load 时已缓存成
   `cann_mega_moe_w13_weight_list/cann_mega_moe_fused_w*_scale_list`
   （`w8a8_dynamic.py:~306-330`），decode 每层只是把 384 个 per-expert tensor 的
   Python list 传给 pybind（`to_list()`）；eager 下是 host 开销，graph 下会被摊掉。
   activation：本模型 `hidden_act=silu`、`swiglu_limit=10.0` ⇒
   `select_mega_moe_activation_kwargs()` 只传 `activation_clamp=10.0`
   （不会走 swigluoai/situ 的额外参数分支），本身没有额外代价。
5. **`mc2_tokens_capacity` 192→16384 的连带效应**：如果 decode 被门槛踢回 MC2，
   MC2 的 `max_num_tokens_per_rank` 从 24 变成 2048，dispatch/combine 的 buffer
   与 metadata 都按更大容量准备（global_bs=0 动态模式缓解，但不是零成本）。
   这是"prefill-only 门槛"自己的代价，需要和 mega 的收益分开算。
6. **draft 也走 mega（如果没排除）**：每 decode step draft 有 3 个 block 的 MoE，
   小 batch 下的 mega 开销会叠 3 次，且 target/draft 两份 symm buffer。

### 2.5 与 `multistream_overlap_shared_expert` 的硬互斥（当前最重要的"叠加性"）

```python
# ascend_config.py:842
if self.enable_fused_mc2 == 1 and self.multistream_overlap_shared_expert:
    self.multistream_overlap_shared_expert = False
```

只要 `additional_config.enable_fused_mc2 ∈ {1,2}`（2 会先降成 1），
不管最后有没有回滚，multistream 都被关。当前 serve.sh 的
`ADDITIONAL_JSON` 里 `"multistream_overlap_shared_expert":true` 是**唯一**与
exp41/exp42 不同的生产开关；这是本次 A/B 的真实自变量。

--------------------------------------------------------------------------------

## 3. 门槛参数（`VLLM_ASCEND_V41_MEGAMOE_PREFILL_MIN_TOKENS`）

### 3.1 我们的 step token 数分布

- decode：`num_tokens = num_reqs × query_len = num_reqs × (1 + K=5) = 6×num_reqs`，
  `max_num_seqs=32` ⇒ **≤192**；图桶列表最大值也正是 192
  （`1,2,4,6,...,184,192`，与 32×6 对齐）。
  DP2 下取 `max(num_tokens_across_dp)`，仍然 ≤192。
- prefill：chunk 上限 `max_num_batched_tokens=16384`（每 DP rank），
  实际 chunk 大小 ∈ {16384（整块）, 1..16383（拒绝/长度余数导致的尾块）}；
  chunked prefill 与 decode 混排时该 step 的总 token 也可到 16384。
- profile/dummy run：用 BAT 量级的 dummy token（16384）跑 eager，会先命中 mega。

### 3.2 阈值表（mega 真正生效时，各阈值覆盖哪些 step）

| MIN_TOKENS | 走 mega 的 decode 桶 | 走 mega 的 prefill | 不满足阈值的回落 | 评价 |
|-----------|----------------------|--------------------|------------------|------|
| 128 | 128,136,144,150,152,160,168,174,176,184,192（≈22–32 并发） | 几乎全部 chunk | <128 → MC2 | ❌ decode 一半桶被污染，且同一 decode 会话里桶切换会换算法，稳定性最差 |
| **256** | 无（全部 ≤192） | chunk ≥256 全走 mega；尾块 1–255 走 MC2 | MC2（capacity 16384 ⇒ 不会 ALLTOALL） | ✅ 激进但安全；多覆盖 256–511 的中小 chunk |
| **512（推荐）** | 无 | chunk ≥512 走 mega | 1–511 → MC2 | ✅ 与 `_MC2_TOKENS_PER_RANK_LIMIT=512` 对齐；512 以下本来就是 MC2 的舒适区 |
| 1024 | 无 | chunk ≥1024 走 mega | 512–1023 走 MC2（baseline 这些尺寸本来走 ALLTOALL） | ⚠️ 中 chunk 换算法，收益/风险不明 |
| 2048 | 无 | chunk ≥2048 走 mega | 512–2047 走 MC2 | ⚠️ 把更多 prefill 让给 MC2，越来越接近"只优化大 chunk"；如 prefill 以整块 16384 为主则影响小，但尾块比例上升 |

**推荐 512**（现状默认值就是 512，不需要改）：
- 严格大于 decode 最大桶 192，decode 零污染；
- 与 MC2 的单 rank 上限 512 对齐，低于阈值的小 chunk 用 MC2 是官方设计意图；
- 256 可以作为第二步微调（预期 TTFT 再降 <0.5%，风险是可测的）；
- **不要 128**（会把 22 并发以上的 decode 桶拖进 mega，且与 Lead 观察到的
  decode 退化叠加）；**不要 ≥1024**（中 chunk 降级 MC2 的变量混入，且门槛越大
  mega 覆盖越小、"prefill-only 优化"意义递减）。

### 3.3 门槛补丁本身的两个副作用（即使 mega 真生效）

1. **capacity 膨胀**（§2.2）：任何被门槛踢回的 step 都不会再选 ALLTOALL。
   如果关心 prefill 尾块，需要单独对比 "capacity 192 基线 vs capacity 16384 但
   门槛排除" 的差异——这需要把 `set_mc2_tokens_capacity` 与选择器解耦（当前没有做）。
2. **multistream 仍被关**（§2.5）：门槛只影响 MoE comm 选择，不影响配置互斥。
   所以"decode 恢复"这个目标在现有代码下**不可达**，除非再做一个
   "prefill-only 模式下允许 multistream" 的配置补丁（见 §6 后续实验）。

--------------------------------------------------------------------------------

## 4. draft 模型（DSpark，3 层）是否会触发 MegaMoe？

事实（远端模型 config + 源码）：
- draft 是 `DSparkDeepseekV41ForCausalLM` → `DeepseekV41DSparkModel`，
  注释写明 "Three serial draft blocks matching the checkpoint's `mtp.*` tree"，
  `num_nextn_predict_layers=3`。
- draft 的 MoE 参数：`dspark_n_routed_experts=128`、`dspark_num_experts_per_tok=3`
  （与 target 384/6 不同），量化与 target 同源（`runner_type == "draft"` 时
  `routed_experts.py:~456` 把 `self.quant_type` 写进
  `draft_model_config.draft_moe_quant_type`，本模型 = W8A8，属于 mega 支持的量化）。
- A3 的选择器 `_select_fused_or_capacity_moe_comm_method()` **不检查**
  `is_draft_model` / `draft_moe_quant_type`（只有 A5 的
  `_select_capacity_and_world_size...` 检查），所以 **mega 生效时 draft 也会走
  FUSED_MC2**，并且因为 `(MoECommType, num_experts, num_local_experts)` 不同，
  会再分配一份 symm buffer（`num_experts=128`、`expert_per_rank=8`、
  `absolute_safe = 2048×16×min(3,8)=98304`，`max_recv=min(65536,98304)=65536`）。
- 如果 prefill-only 门槛生效：draft 的 **decode 步**（tokens ≤192×? draft 每 step
  的 token 数同样是 num_reqs×(1+K)? draft 只跑 1 次 forward 生成 K 个候选，
  token 数 ≈ num_reqs）会被阈值排除；但 **draft prefill**
  （`is_draft_model_prefill`，处理 prompt token）仍会 ≥512 而命中 mega。

结论与建议：**应当排除 draft**。理由：
1. decode 延迟敏感，draft 是三段串行的小 MoE，mega 在小 M 下效率最差；
2. 两份 symm buffer 要多占 1.5–2.5GiB/rank 量级通信内存；
3. 排除的风险很低：W8A8 mega 权重准备保留原始 `w13_weight/w2_weight`
   （§0310 说明），降级 MC2 安全。
=> 补丁 `scripts/patches/experimental/0310-megamoe-exclude-draft.patch`，
   env `VLLM_ASCEND_V41_MEGAMOE_EXCLUDE_DRAFT=1`（默认 0，不改变现状）。
   注意它与 prefill-only 门槛是**互补**的：门槛排除 decode、这个补丁排除 draft 的
   prefill 阶段；两者都开时 draft 全程 MC2。

--------------------------------------------------------------------------------

## 5. 叠加性矩阵（阈值 × draft 排除 × limit_core_num）

前提：以下所有格子都建立在 **0311 绕过形状白名单成功、MegaMoe 真能跑** 的假设上。
在当前代码下全部是"不可达"，只能作为实验规划。

记号：LC = 0202 `limit_core_num`（默认开，MAIN_AIV=16/AUX_AIV=32/AIC=24）；
D- = 0310 排除 draft；D+ = draft 也走 mega。

| 阈值 | D- + LC 开 | D- + LC 关 | D+ + LC 开 | D+ + LC 关 | 备注 |
|------|-----------|-----------|-----------|-----------|------|
| 128 | decode 高桶 mega；LC 只作用于 attention 流，与 mega 无直接冲突；预期 decode ms/step **+1~3%**（mega 小批 + 丢 multistream），TTFT 与 512 接近 | 同左但少了 LC 的 +1~4% 单流收益 ⇒ decode 更差 | 再叠加 draft 3 层小批 mega ⇒ decode **+3~6%**，风险最高 | 最差组合 | ❌ 不推荐 |
| 256 | decode 全 MC2 ⇒ decode 退化只剩"丢 multistream"（−5%）与 capacity 膨胀（≤1%）；prefill 覆盖最大 ⇒ TTFT 最优候选 | decode 同上，prefill 少 LC 收益 | draft prefill 也 mega（3 层 × 少量 token）⇒ prefill 收益被 draft 摊薄 | 同左 | ⚠️ 激进备选 |
| **512** | decode 全 MC2；prefill ≥512 mega；draft 全 MC2（若 D-）；**推荐组合** | 推荐对照组（验证 LC 与 mega 是否叠加） | draft prefill 走 mega，decode 仍 MC2；风险中等 | 同左 | ✅ 首选 |
| 1024 | decode 全 MC2；512–1023 chunk 降级 MC2（baseline 是 ALLTOALL）⇒ TTFT 可能反而略差 | 同左 | 同 512+D+，但 mega 覆盖更小 | 同左 | ⚠️ 仅用于定位"mega 在大 chunk 的收益" |
| 2048 | 只有 ≥2048 的 chunk 走 mega；prefill 收益最小，MC2 占比大 | 同左 | 同左 | 同左 | ❌ 除非确认 prefill 全是 16384 整块 |

预期数值（相对 exp40 基线，仅供实验排序，不作为承诺）：
- TTFT：如果 mega op 真能跑，乐观 −5~10%（整块 16384 的 prefill 是它唯一的甜点区）；
  但注意 **exp29/41 的"−20~25% TTFT"已被证伪**，不要把那个数当预期。
- decode ms/step：门槛 ≥256 时仍会因 multistream 互斥而 −3~5%；
  只有再解耦 multistream 才可能回到基线。
- 风险等级：128 最高（decode 路径切换 + 图桶异构）；512/1024 中等
  （symm buffer 分配时序、精度、HCCL abort）；0311 本身是最高风险项
  （算子 tiling 可能直接拒绝 4608 的 intermediate_hidden）。

### 5.1 与其他旋钮的源码级相互作用

| 旋钮 | 与 MegaMoe 的关系（源码依据） | 建议 |
|------|------------------------------|------|
| `0202 limit_core_num` | 作用于 attention/indexer 的主/辅流（`dsa_v41.py` 的 `limit_core_num(aic,aiv,stream)` 作用域），MegaMoe 调用发生在 MoE 层、不在该 scope 内 ⇒ **无直接耦合**。但两者都吃 AIV/AIC 资源，收益可能部分重叠（都减少 vector 核争用）；不会互相破坏 | 保持默认开；在 mega 实验里固定 LC 不变，单独一格对照 LC 关 |
| `multistream_overlap_shared_expert` | **硬互斥**（`ascend_config.py:842`）：`enable_fused_mc2∈{1,2}` ⇒ 强制 False。这是当前 decode −5% 的唯一原因 | 生产保持 `enable_fused_mc2=0` 让 overlap 保留；如要 mega，需额外做一个"MegaMoe 下保留 overlap"的补丁（未提供，风险高） |
| `HCCL_BUFFSIZE=1024` | MegaMoe 用 `CommContextManager` 的 custom CCL buffer（`customCclBufferSize` 由算子公式算），**不吃** `HCCL_BUFFSIZE`；普通 MC2 才吃。mega 启用后 capacity 192→16384 会让 MC2 回退路径的缓冲区按 2048/rank 准备，若再把 HCCL_BUFFSIZE 调小可能触发 buffer 不足 | 保持 1024；不要为 mega 调小 HCCL_BUFFSIZE |
| DP2 / `should_skip_allreduce` | `is_mega_moe_supported()` 时强制 `should_skip=False`（`utils.py:1328`），要求 DP 均匀 token + 每步 allreduce；我们本来就是 False，无额外损失。PD 分离的 decode 节点会失去 skip 优化 | 本拓扑无影响；其它拓扑需重算 |
| `enable_prefill_mc2` | 与 mega 是两条独立路径；mega 生效时 capacity 分支已被 `use_mega_moe` 抢占，`enable_prefill_mc2` 不再改变 capacity | 保持 False |
| `enable_fused_mc2=1`（老 `dispatch_ffn_combine`） | 同一配置位；A3 分支 `enable_fused_mc2==1 and ep<=32` 直接返回 FUSED_MC2，同样触发 multistream 互斥，且不支持 MiniMax M3 SwiGLU-OAI。对本模型（silu）理论可用，但上限 512/rank、无条件 prefill/decode 不分 | 不建议回退到 1；要试就试 0311 |

--------------------------------------------------------------------------------

## 6. 验证口径（下次真要测 MegaMoe 时按这个判）

### 6.1 启动日志断言（缺一不可）

```bash
sv=/cache/exp/logs/serve_dsv41w8a8rc2-<ts>.log
grep -c "MegaMoe is not supported for this model config" $sv   # 必须 == 0
grep -c "CANN MegaMoe sym-buffer alloc"                     $sv   # 必须 >= 1（D- 时 target 一份；D+ 时两份）
grep -c "MegaMoe received .* tokens per rank, but its symmetric buffer" $sv  # 必须 == 0
grep -c "Shared expert split computation validation passed" $sv   # 判断 multistream 是否仍开着
grep -c "enable_fused_mc2 and multistream_overlap_shared_expert cannot be enabled" $sv  # 期望逻辑（mega 时必现）
```

另外看 worker 的 `Actual usage: ... GiB for weights, ... non-torch memory`：
mega 真启用时 `non-torch memory` / `torch reserved` 应比基线增加 symm buffer 量级
（估算 1.5–2.5GiB/rank，D+ 再加一份）；graph capture 时间也应比 73–75s 更长。

### 6.2 指标口径

- TTFT：`single_cold`（同一 prompt、无 prefix cache 干扰）+ `single_hot` 三次 run 取中位数；
  **只用一次 single_hot 会被 baseline 方差骗**（exp36 249 vs exp40 298）。
- decode：优先用引擎侧 `ms_per_step`（`/cache/exp/json` 里的字段），
  其次 `single_cold.decode_tps` 与 `dual c=2.stream`；
  `single_hot.decode_tps` 受 acceptance 波动影响大，仅作参考。
- 容量/并发：`dual c=16 / c=64` 的 aggregate，用于确认没有把高并发路径搞退化。
- 精度：mega 走 INT8 dispatch（mode=2），必须补一轮 greedy 输出对比
  （同一 seed/prompt 的 token 序列或 logprob 差异），不能只看性能。

### 6.3 判定

- 若日志断言 1/2 不满足 ⇒ MegaMoe 没跑，任何性能变化都归因到别的开关
  （本次就是 multistream）。
- 若 symm buffer 日志出现但 decode 仍 −5% ⇒ 大概率还是 multistream 互斥，
  需要先做"MegaMoe 下保留 multistream"的实验，否则无法归因。

--------------------------------------------------------------------------------

## 7. 结论与建议（给 Lead）

1. **不要在生产保留 `additional_config.enable_fused_mc2=2`。** 它不但没启用 MegaMoe，
   还会顺带关掉 `multistream_overlap_shared_expert`，实测单流 decode −5%，
   TTFT 无真实收益。当前生产应回到 exp40 的默认（`enable_fused_mc2` 不设 / 0）。
2. **exp41/exp42 的定位改写：它们是"multistream overlap 关闭"的对照实验**，
   不是 MegaMoe A/B。exp41 与 exp29 数值高度一致（cold decode 109.2 vs 103.2、
   dual c=2 stream 97.1 vs 94.8），exp42 与 exp41 又几乎完全相同
   （cold decode 109.8 vs 109.2、dual c=2 stream 97.9 vs 97.1），
   进一步佐证三者是同一配置（prefill-only 门槛在 mega 不可达时为死代码）。
3. **MegaMoe 在本模型上结构性不可用**（`moe_intermediate_size=2304` 被 A3/A5
   两套白名单同时拒绝）。如果还想给它一次机会：
   - 先打 `0311-megamoe-allow-unsupported-shape.patch`（env 默认关），
     单次 smoke 启动验证算子是否接受 `hidden=5120 / intermediate_hidden=4608`；
     大概率会在 tiling/profile 阶段报错——那样就此结案，不值得再投入。
   - 如果能跑，再打 `0310` 排除 draft，阈值保持 **512**（备选 256），
     并按 §6 的日志断言确认 mega 真的在位。
4. **真正的 decode 收益方向不在 MegaMoe。** 本次 A/B 反而证明了
   shared-expert multistream overlap 值 ~5% 单流 decode：
   - 若将来做 mega，必须同时解决 `ascend_config.py:842` 的互斥（目前无补丁，
     因为 mega 在 prefill eager 阶段可能与 aux stream overlap 冲突，需要算子侧确认）；
   - 若不做 mega，这 5% 就是"保持现状"的收益，别为了 TTFT 噪声把它丢掉。
5. 门槛补丁（Lead 已写）本身没有问题，但在 mega 不可达时是**死代码**；
   建议把它作为"未来 mega 可用时的默认策略"保留在实验序列里，
   `MIN_TOKENS` 固定 512，并在文档标注依赖 `enable_fused_mc2=2` 真正生效。

--------------------------------------------------------------------------------

## 8. 附录

### 8.1 源码索引（远端只读，行号为近似值，Lead 的补丁会让文件偏移 ~+9 行）

| 文件 | 位置 | 内容 |
|------|------|------|
| `ascend_forward_context.py` | L35–47 | `_MEGA_MOE_TOKENS_PER_RANK_LIMIT=16384`、Lead 的 prefill-only 门槛常量 |
| `ascend_forward_context.py` | L~87 | `use_cann_megamoe()`（enable_fused_mc2==1 才 True） |
| `ascend_forward_context.py` | L~252–283 | `set_mc2_tokens_capacity()`（mega ⇒ capacity=BAT=16384） |
| `ascend_forward_context.py` | L~335 | `_select_fused_or_capacity_moe_comm_method()`（A3 策略） |
| `ascend_config.py` | L608 | `enable_fused_mc2: int = 0` |
| `ascend_config.py` | L667–676 | `=2` 的 rollback：`_MEGA_MOE_SUPPORTED=find_spec(...); enable_fused_mc2=1` |
| `ascend_config.py` | L842–852 | multistream 互斥 → 形状检查失败 → `enable_fused_mc2=0` |
| `ascend_config.py` | L1052–1090 | `_is_megamoe_supported_by_config()`（2304 % 512 != 0 ⇒ False） |
| `ascend_config.py` | L600 | `mega_moe_max_tokens: int = 65536` |
| `ops/fused_moe/moe_comm_method.py` | L289–615 | `FusedMC2CommImpl`：symm buffer 生命周期、mega 调用、dispatch_ffn 回退 |
| `ops/fused_moe/moe_comm_method.py` | L386–467 | `_init_mega_moe_symm_buffer()`（尺寸公式） |
| `ops/fused_moe/moe_comm_method.py` | L469–560 | `_apply_cann_mega_moe()`（权重列表/mask/activation/kwargs） |
| `ops/fused_moe/moe_utils.py` | L44–60 | `_QUANT_SETTING_MAP`（W8A8 → INT8 mode=2） |
| `ops/fused_moe/moe_utils.py` | L144–164 | `load_cann_mega_moe_ops()`（提前加载 `comm_context_op_builder`） |
| `ops/fused_moe/moe_utils.py` | L167–254 | `select_mega_moe_activation_kwargs()`（silu 只用 activation_clamp） |
| `ops/fused_moe/token_dispatcher.py` | L207–224 | `max_num_tokens_per_rank` / `global_bs` 派生 |
| `utils.py` | L1304–1360 | `should_skip_allreduce_across_dp_group()`（mega ⇒ 强制 False） |
| `quantization/methods/w8a8/w8a8_dynamic.py` | L306–330 | mega 权重视图缓存（保留原始 weight） |
| `ops/fused_moe/shared_experts.py` | L125、L460–630 | multistream overlap 的 aux stream 调度 |
| `cann_ops_transformer/ops/mc2/mega_moe/mega_moe.py` | L273–588 | `SymmBuffer` / `get_symm_buffer_for_mega_moe`（custom CCL buffer） |
| `cann_ops_transformer/ops/mc2/mega_moe/graph_convert_mega_moe.py` | — | torchair GE converter（graph mode 可捕获） |

### 8.2 本次新增的实验补丁

- `scripts/patches/experimental/0310-megamoe-exclude-draft.patch`
  env `VLLM_ASCEND_V41_MEGAMOE_EXCLUDE_DRAFT=1`，默认关；
  依赖 Lead 的 prefill-only 门槛补丁已在树里。
- `scripts/patches/experimental/0311-megamoe-allow-unsupported-shape.patch`
  env `VLLM_ASCEND_V41_MEGAMOE_ALLOW_UNSUPPORTED_SHAPE=1`，默认关；
  作用于干净的 `ascend_config.py`。
- 说明见 `scripts/patches/experimental/README-megamoe.md`。
  env 在 spawn worker 里可能不继承，实际启用时优先把默认值 pin 成 `"1"`。

### 8.3 只读复核命令

```bash
SSH='ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null new_rl'
$SSH 'grep -c "MegaMoe is not supported" /cache/exp/logs/serve_dsv41w8a8rc2-20261112_142053.log'
$SSH 'grep -c "CANN MegaMoe sym-buffer alloc" /cache/exp/logs/serve_dsv41w8a8rc2-20261112_142053.log'
$SSH 'grep -n "_MEGAMOE_PREFILL_MIN_TOKENS\|_is_megamoe_supported_by_config" \
  /cache/dsv41-rc2-rootfs/vllm-workspace/vllm-ascend/vllm_ascend/ascend_forward_context.py \
  /cache/dsv41-rc2-rootfs/vllm-workspace/vllm-ascend/vllm_ascend/ascend_config.py'
```
