# 融合/算子优化方案：F1 / F2 / F4 / F6 + 上游融合开关盘点（2026-11-12）

> 作者：fusion-dev（task-4）｜**只读远端源码，未启停任何引擎、未做压测**
> 源码口径：`/cache/dsv41-rc2-rootfs/vllm-workspace/vllm-ascend`（= 本地镜像副本
> `rootfs/vllm-workspace/vllm-ascend`，git `8fde6d4e3` + 已应用主序列 0001/0002/0004/0005/0202/0101）。
> 下文行号均指**这份已打补丁的树**。
> 参考数据：repo1 文档快照 `/tmp/refs/a/docs/`（`FUSION-CANDIDATES-20261007.md`、
> `FRESH-CONC1-TOP-OPS-AND-HCPRE-BREAKDOWN-20261007.md`、`QNORM-FUSE-A-B-20261008.md`、
> `MULTI-OP-FUSION-FEASIBILITY-20261008.md`、`SINGLE-STREAM-LEVERS-EXHAUSTED-20261007.md`）。
> **生产参数不动**：BAT=16384 / max_num_seqs=32 / 图桶≤192 / W8A8 全部保持。

---

## 0. 一页纸结论

1. **上游"融合开关"看起来已经开了，但对 V4.1 路径全部无效**：
   日志 `Enabled custom fusions: norm_quant, act_quant` 只是 vllm `CompilationConfig`
   的两个字段为真（`vllm/config/vllm.py:154-176`），而我们这条路的注意力整体被
   **自定义算子边界 `dsa_v41_forward`**（`attention/dsa_v41.py:84-110`）挡住，
   FX/inductor pass 根本看不到里面；且在 model 层能匹配的模式（`npu_add_rms_norm_bias`
   + `torch.ops.vllm.quantize` / q-k-v split / `x*scale+y`）**V4.1 一个都不用**。
   ⇒ **"再开一个上游开关"这条路没有收益**，真正的 norm+quant 融合必须回到模块级替换。
2. **能纯 Python 落地的融合，天花板远低于 F1/F2/F4/F6 的族统计值**：
   repo1 的 F 族上界是 **≈1.76 ms（其 25.46 ms 步长的 7%）**，但那是"把多个 kernel 合成一个"
   的收益；Python 只能**删掉本来就不该算的算子**（冗余/死代码），不能把两个真 kernel 合并。
   我们盘出来的纯 Python 可回收：**≈0.2~0.35 ms/步 ≈ 0.5~0.9%（按 40 ms/步）**。
3. **本次交付 3 个补丁（默认 env=0，行为与上游逐位一致）**：
   `0301` q_norm+wq_b 融合（repo1 实测 −0.62%/步）、`0302` 槽位链死代码消除、
   `0303` o_proj padding 零填充消除。三者在副本树上 `git apply` 顺序通过、`py_compile` 通过。
4. **F4 / F6 的"融合"本体不可纯 Python 完成**：gating 已是 `MoeGatingTopKHash`、
   routing/unpermute 已是 CANN 大算子；indexer 三件套里两个已是 Triton、一个是 CANN QLI。
   再合需要 AscendC（本镜像可直接照抄 `rms_norm_dynamic_quant` 的 proto/host/kernel 模板）。
5. **最大的单项 F2 不是 Python 能拿的**：每层一次 `output[...] = self.wo_b(...)`（`attention/dsa_v1.py:1837`）＝ repo1 profile 里的 `InplaceCopy_TensorMove`
   43.5 次/步 × 5.12 µs ≈ **0.256 ms**；`npu_quant_matmul` 没有 `out=`，只能靠算子 epilogue
   或图边界改造，**不在本次 patch 范围**（见 §5 否决项）。

---

## 1. 数据口径与折算方法（先看这个再看数字）

| 量 | repo1（参考仓库） | 我们（本部署） |
|---|---|---|
| 单步 | 25.46 ms（纯 conc=1 profile，K≈1） | **≈40.2–41.9 ms**（`extra-info/optimization-report-deepdive.md` §1） |
| 层数 | 43（40 主层 + 3 draft）| 同 |
| 算子数 | 2,821/步 | 结构相同（未本地 profile，推断同量级） |
| 拓扑 | W4A8 + TP8/EP8（8 die） | W8A8 + DP2×TP8/EP16（16 die） |
| 融合族上界（F1+F2+F4+F6） | **1.76 ms = 7.0%**（`SINGLE-STREAM-LEVERS-EXHAUSTED` §3） | 绝对 µs 可平移；**占比要 ÷40 而不是 ÷25.46** |

**折算规则**：按"次数/步 × 单次 µs"平移（层数相同、同型号 A3 ⇒ 单 kernel 固定成本接近），
得到我们口径的 **Δms/步**，再除以 40 ms 得占比。**但** repo1 明确写了：
算子级收益**不能**外推到图模式 E2E（`hc_pre_norm` standalone −0.459 ms → E2E +0.359 ms，
`FRESH-CONC1-TOP-OPS...` §4.2）。所以下面每个数字都标了"上界/实测"。

---

## 2. 上游已有融合开关盘点（结论：都开着，但都打不到我们）

日志证据：`logs/serve_dsv41w8a8rc2-20261101_222549.log:110-111,1053,1425-1426`
`[compilation.py:336] Enabled custom fusions: norm_quant, act_quant`。

| 开关 | 默认 | 注册位置 / 生效条件 | 在 V4.1 路径上是否生效 | 判定 |
|---|---|---|---|---|
| `fuse_norm_quant` → `AddRMSNormQuantFusionPass` | True（`vllm_ascend/ascend_config.py:127`） | `compilation/graph_fusion_pass_manager.py:52-64`；需 `HardwareCapability.GRAPH_NORM_QUANT_FUSION`（A3 在标准能力集里，`device/hardware_profile.py:251-280`） | ❌ pass 只匹配 `torch.ops._C_ascend.npu_add_rms_norm_bias` + `torch.ops.vllm.quantize`（`passes/norm_quant_fusion_pass.py:42-92`，静态 FP8 路径）。V4.1 W8A8-dynamic 走 `RMSNorm` / `npu_dynamic_quant` / module 级 `quantize()`，**永不匹配**；且注意力在自定义算子边界内（`dsa_v41.py:84-110`），FX 看不见 | **无效**（repo1 profile 里 `RmsNorm` 仍 140 次/步就是证据） |
| `fuse_qknorm_rope` → `QKNormRopeFusionPass` | True（`ascend_config.py:128`） | `graph_fusion_pass_manager.py:65-69`（无硬件门） | ❌ 模式要求 `qkv.split(...)` + q/k 两次 `npu_rms_norm` + `npu_rotary_embedding`（`passes/qknorm_rope_fusion_pass.py:57-81`）；DSA/MLA-V4.1 **没有 qkv split** | **无效** |
| `fuse_muls_add` → `MulsAddFusionPass` | True（`ascend_config.py:129`） | `graph_fusion_pass_manager.py:70-73`；需 `GRAPH_MULS_ADD_FUSION`（A3 有） | ⚠️ 模式是 `x*scale + y`（`passes/muls_add_pass.py:31-85`）；V4.1 热路径的可编译区域里未见稳定匹配（hc/attention/MoE 全是自定义算子），且 `scale` 是 pattern 实例常量，实际命中面很窄 | **基本无效**（可留观，不值得单独立项） |
| `fuse_act_quant` / `fuse_norm_quant`（vllm 侧 CUDA pass：`vllm/compilation/passes/fusion/{act_quant,rms_quant}_fusion.py`） | 由 `enable_norm_fusion`/`enable_act_fusion` 决定（`vllm/config/vllm.py:154-176`） | vllm 的 `PostGradPassManager` 路径 | ❌ vllm-ascend **替换成自己的 `GraphFusionPassManager`**（`graph_fusion_pass_manager.py:27-73`），这三个 CUDA pass 根本没注册 | **无效** |

**结论**：可开的开关已经全开（且默认就是 True）；日志里的 "Enabled custom fusions" 只是
**字段为真**，不代表有 pass 命中。⇒ 不要在这条路上继续找开关；
真正能做的是 §4 的 `0301`（模块级把 norm+quant 换成官方融合算子）。

补充：树里**已经做过**的模块级融合（别再重复投入）：

| 已融合项 | 位置 |
|---|---|
| `npu_hc_pre_v3` / `npu_hc_post`（hc 前后处理各 1 kernel） | `models/deepseek_v41/model.py:863-880` |
| `rms_norm_cast`（norm 一次出 bf16+fp32，喂 MoE 路由） | `model.py:849-857`、`:917` |
| `GroupedMatmulSwigluQuantV2`（MoE w1/w3 + SwiGLU + quant） | 量化方法内部（`quantization/methods/w8a8/w8a8_dynamic.py:407-529`） |
| `npu_sparse_flash_mla`（SMLA 一体） | `attention/dsa_v41.py:648-720` |
| MC2 dispatch/combine（decode MoE 通信+permute 一体） | `ops/fused_moe/token_dispatcher.py:189-...`、`moe_comm_method.py:229-260` |

---

## 3. 逐项分析（F1 / F2 / F4 / F6）

### 3.1 F1 位置/槽位链 —— 纯 Python 只剩"死代码"，重构需 Triton/CANN

**源码位置**

| 环节 | 文件:行 |
|---|---|
| 槽位映射准备（按 KV group 每步多次） | `attention/dsa_v41.py:1058-1195`（`build()` 内） |
| 压缩槽位规则 `compressed_slot_mapping`（`>=0` / `(s+1)%ratio` / `//` / `where`） | `attention/dsa_v41.py:223-231`，调用点 `:1090-1096` |
| Triton 单发路径 `build_packed_cache_slot_mapping` | `ops/triton/packed_cache_slot_mapping.py`，调用点 `dsa_v41.py:1095-1145` |
| eager 回退链（`ClampMin/Cast/FloorDiv/Remainder/Where` ×2 列 + flat） | `dsa_v41.py:1146-1193` |
| int64 槽位缓冲（`slot_mapping` / `flat_slot_mapping`） | `dsa_v41.py:811-813` |
| `positions.long()`（每步 1 次 Cast） | `dsa_v41.py:998` |
| 每层 `pad_sparse_indices().contiguous().int()` | `dsa_v41.py:273-278`（调用 `:684`） |
| 写 cache 时的 `values.to(cache.dtype).contiguous()` | `dsa_v41.py:255-271` |
| 基础 slot mapping（Triton 单发，已下沉） | `worker/block_table.py:151-195`、`ops/triton/compute_slot_mapping.py:12-100` |

**repo1 数据**：F1 族 469.3 次/步、自身 0.981 ms、**真实暴露 0.356 ms（其步长 1.4%）**；
其中 `SelectV2 48×1.92µs / Less 46×1.38 / Abs 45×1.28 / Neg 45×1.50 / FloorMod 24×2.34 /
Arange 6×10.12`。**折算我们口径：≈0.36 ms/步 ≈ 0.9%（÷40 ms）**，
但 repo1 同时指出：F1 与 F2 在**步尾同一条 stream 47 尾链**上交织，
`ViewCopy[16384]` 那 0.193 ms 是否在关键路径**尚未确认**（`FRESH-CONC1...` §5.5.2）。

**我们已定位的一处真冗余（→ patch 0302）**：
`dsa_v41.py:1076-1096` 里 `compressed_slot_mapping()` 在**进入 Triton 分支之前无条件执行**，
而 Triton kernel 明确要求吃 **raw** slot mapping（注释 `:1099-1105` 自己写了
"pre-compressing would apply `slot // ratio` twice"）。也就是说只要
`VLLM_ASCEND_V41_TRITON_SLOT_MAPPING=1`（主序列已 pin 成 1），这条
`>=0` / `(s+1)%ratio` / `//` / `where` 链（≈6-7 个 kernel）就是**纯死代码**，
每个 KV group 每步白跑一次（`build` 每步 13 组；KV group 数=13，见
`extra-info/code-opportunities.md` §1）。

**可实现性**：Python ✅（0302）；把剩余位置链再下沉 1 个 kernel = Triton/CANN，非本次范围。
**精度风险**：0302 **零**（删的是未被读取的计算）。
**验证**：kernel 计数里 `SelectV2/FloorDiv/Remainder/Less/Eq/And` 应各减少 N 次/步；
`bneck hp` p50 前后对比；冒烟 + 结构化任务逐字一致。

### 3.2 F2 dtype/搬运链 —— 最大两项一项要算子、一项已被上游做过

**源码位置**

| 环节 | 文件:行 | 说明 |
|---|---|---|
| ★ `output[...] = self.wo_b(o_proj_input)` | `attention/dsa_v1.py:1837` | 每层 1 次 `TensorMove`；repo1 profile 里 `InplaceCopy_TensorMove` **43.5 次/步 × 5.12 µs = 0.256 ms**（推断：43.5 ≈ 43 层，与『每层一次 o_proj 回拷』吻合，待 kernel 邻接核对） |
| ★ padding 补零 + 拷贝 | `attention/dsa_v41.py:462-469`（`_project_output`） | 每层 `new_zeros`（Fill）+ 一次 copy；**0303 去掉零填充** |
| topk 结果回写 | `dsa_v41.py:641-646` | 只在 index-source 层（8 层），非 43 |
| int64 flat slot 的 `where` 写入 | `dsa_v41.py:1186-1192` | 与 F1 同一条链 |
| `expert_tokens.to(torch.int64)` / `probs` cast | `ops/fused_moe/token_dispatcher.py:504`、`:523` | 每层 1 次 Cast（43/步）；下游 GEMM 需要 int64/bf16，改 dtype 有风险 |
| indexer `weights...float()*scale` 再 `.to(float16)` | `models/deepseek_v41/indexer.py:188`、`:270` | 只在 index-source 层（8 层） |
| `positions.long()` | `dsa_v41.py:998` | 每步 1 次 |

**repo1 数据**：F2 族 491 次/步、暴露 0.888 ms（3.6%）；**Top16 只有 0.620 ms（2.4%）**，
其中 `ViewCopy[16384]` 0.193 ms 与 `TensorMove` 0.256 ms 是两项大头
（`FRESH-CONC1...` §5.5）。**折算我们口径：≈0.62~0.89 ms ≈ 1.6~2.2%（÷40）**，
但其中 **0.256 ms 属于 `wo_b` 输出回拷，Python 无法消除**（`npu_quant_matmul` 无 `out=`）。

**可实现性**：Python ✅ 只有零填充/无效拷贝这类边角（0303 ≈ 43×1.16 µs）；
`wo_b` 回拷要"算子带 epilogue 直接写 output"或"取消 `mutates_args` 图边界"→ AscendC / 图改造。
**精度风险**：0303 零（padding 行本来就是被 runner 丢弃的桶填充行）。
**验证**：`Cast/Fill/ZerosLike/TensorMove` 计数；逐字一致；padding 桶（BAT 覆盖的图桶）冒烟。

### 3.3 F4 gating+routing —— 两个组件**都已经是融合算子**，合并需 AscendC

**源码位置**

| 环节 | 文件:行 |
|---|---|
| V4.1 实际 gating（`sqrtsoftplus` 路由）→ **单个 CANN 算子** | `ops/fused_moe/router/fused_topk_router.py:204-224`（`torch.ops._C_ascend.moe_gating_top_k_hash`） |
| 非 sqrtsoftplus 的通用 gating（也是 CANN 算子） | 同上 `:227-243`（`DeviceOperator.moe_gating_top_k`） |
| 纯 eager 的 grouped-topk 参考实现（仅当不走上面两个算子时才执行） | `ops/fused_moe/router/grouped_topk_router.py:61-125`（softmax/sigmoid→+bias→view.max→topk(group)→zeros_like+scatter_→expand→masked_fill→topk→to(int32)/sum/mul） |
| routing（init routing / dispatch） | `ops/fused_moe/token_dispatcher.py:492-505`；`:769-780`（`npu_moe_init_routing_v2`） |
| dispatch/combine 融合（MC2） | `token_dispatcher.py:189-...`；`moe_comm_method.py:229-260`；选择器 `ascend_forward_context.py:296-333`（A3 = `FUSED_OR_CAPACITY`，decode ≤ capacity ⇒ MC2） |
| unpermute | `token_dispatcher.py:520-524`（`npu_moe_token_unpermute`） |

**repo1 数据**：F4 族 129 次/步、自身 1.690 ms、**暴露 0.384 ms（1.6%）**；
三项 `aclnnMoeInitRoutingV3 43×14.5µs` / `MoeGatingTopKHash 43×13.73µs` /
`MoeTokenUnpermute 43×10.84µs`；repo1 明确"**可合的是 gating+routing（86→43）**，
unpermute 必须等 experts 之后"。**折算：≈0.38 ms ≈ 0.96%。**

**可实现性**：Python ❌（两个都是单算子，删不掉；把 gating 的输出直接喂 routing 需要在
CANN 内部接线）→ **AscendC**（把 `moe_gating_top_k_hash` 的 topk_ids/weights 输出
接到 routing 的 permute 输入，省一次 GM 往返 + 一次 launch）。
**精度风险**：中（routing 语义/边界 sentinel）。
**验证**：`MoeGatingTopKHash` 次数 43→0、`MoeInitRouting*` 43→43 但时长下降；
router 输出（topk_ids/weights）逐位一致；MoE 输出逐字一致。

⚠️ 注意：由于 MC2 选择器在 decode 下走 MC2（`FUSED_OR_CAPACITY`），
必须先确认本部署 decode 实际走的是 MC2 dispatch 还是 AllGather routing——
两条路的算子名不同，但**结论相同**（都已在 CANN 内融合，Python 层无事可做）。

### 3.4 F6 indexer 三件套 —— 两个 Triton + 一个 CANN，已在辅助流上重叠

**源码位置**

| 环节 | 文件:行 |
|---|---|
| query 投影 / RoPE / 量化入口 | `models/deepseek_v41/indexer.py:181-193`（`project_query` / `apply_query_rope` / `quantize_query`） |
| 量化 query（Triton 单发，按 head 做 int8+fp16 scale） | `ops/triton/quantize_indexer_query.py:12-52`（kernel）、`:55-85`（dispatcher） |
| QLI 本体（CANN，score 序 topk） | `models/deepseek_v41/indexer.py:297-311`（`npu_quant_lightning_indexer_v3`） |
| 结果过滤/排序（Triton 单发，`tl.extra.cann.extension.sort`） | `ops/triton/prepare_indexer_indices.py:16-72`（kernel）、`:74-140`（dispatcher）；调用 `indexer.py:313-319` |
| 两条辅助流重叠（weights 投影 / query 量化） | `attention/dsa_v41.py:422-459`（`_prepare_indexer_inputs`、`_quantize_indexer_query`） |
| 每层 `weights.to(float16)` | `indexer.py:270` |

**repo1 数据**：F6 族 24 次/步、自身 0.655 ms、**暴露 0.403 ms（1.6%）**；
其中 `QuantLightningIndexerV2` 8×48.93µs = 0.359 ms **本体已优化过**（99.3→50.3 µs），
`_prepare_indexer_indices_kernel 8×19.56µs`、`_quantize_indexer_query_kernel 8×17.20µs`。
**折算：≈0.40 ms ≈ 1.0%**，但**可回收部分只有 0.15 ms 量级**（两个小 kernel），
且它们已在 aux stream 上与主流的 cube 工作重叠（`dsa_v41.py:422-459`）。

**可实现性**：Python ❌（两个 kernel 都是单发 Triton，合成 QLI 的 prologue 需 AscendC）。
**精度风险**：中（QLI 输入布局/scale 契约）。
**验证**：`QuantLightningIndexerV2` 次数不变但时长下降；topk indices 逐位一致；
长文针（60K/74K/150K）全过。

⚠️ 另有 3 个 AICPU metadata 算子很大（`SparseFlashMlaMetadata` 236 µs×3、
`QuantLightningIndexerV2Metadata` 186 µs×2、`SparseAttnSharedkvMetadata` 213 µs×2），
但 repo1 已实测其**暴露只有 0.03 ms**（被其它流覆盖）⇒ **不要再投入**。

---

## 4. 本次交付的补丁（`scripts/patches/experimental/`，全部默认 env=0）

> 同目录另有 MegaMoe 那条线的 `0310/0311`（`README-megamoe.md`），**故意不在本线的
> `series.conf` 里登记**，两条线互不牵连。

应用（不动主序列；custom 单组件模式）：

```bash
SRC=/cache/dsv41-rc2-rootfs/vllm-workspace/vllm-ascend \
  PATCH_DIR=$PWD/scripts/patches/experimental \
  bash scripts/apply_patches.sh apply
# 回滚
SRC=/cache/dsv41-rc2-rootfs/vllm-workspace/vllm-ascend \
  PATCH_DIR=$PWD/scripts/patches/experimental \
  bash scripts/apply_patches.sh revert --force
```

> 已在副本树上验证：三补丁按 `series.conf` 顺序 `git apply --check` 通过、`py_compile` 通过。
> **默认全部关闭**（env 缺省 `0` ⇒ 与上游逐位一致）。

### 4.1 `0301-qnorm-wq-b-fuse.patch` —— `q_norm` + `wq_b.quantize` 融合 【F3/F2】

| 项 | 内容 |
|---|---|
| env | `VLLM_ASCEND_V41_QNORM_FUSE`（默认 `0`） |
| 改动点 | `attention/dsa_v41.py::multistream_preprocess`（`q_norm`→`wq_b.quantize` 两行换成 `torch.ops._C_ascend.npu_rms_norm_dynamic_quant`）；新增模块级 gate（同 0202 的写法） |
| 前置既有实现 | `attention/dsa_v1.py:1958-1968` 已在上游 V1 路径用同一融合算子；`torch_binding.cpp:1597-1630` 的 schema `(x, gamma, smooth_scale=None, beta=None, epsilon=1e-6) -> (y_int8, scale_fp32)` |
| 覆盖面 | 43 层里 **35 层**（8 个 index-source 层留给 indexer 用 bf16 `qr`，见 `dsa_v41.py:602-611`）；门控另含 `not wq_b._has_communication and wq_b._is_w8a8_dynamic` |
| 预计收益 | **repo1 实测 −0.149 ms/步 = −0.62%**（`QNORM-FUSE-A-B` §0）；kernel 计数 `RmsNormDynamicQuant 3→35`、`RmsNorm 140→108`。换成我们 40 ms 步长 ⇒ **≈0.10~0.15 ms/步 ≈ 0.25~0.40%**（绝对值平移，占比变小） |
| 可实现性 | Python ✅（无需新算子） |
| 精度风险 | **中**：官方融合算子**非逐位等价**（repo1 F2 实验实测 4~5% 元素差 1 LSB）；必须过"结构化任务逐字一致 + 100% 正确"分级门 |
| 启用方式 | 推荐像主序列一样 **pin 默认值**（spawn worker 不继承 `os.environ`）：<br>`sed -i 's/"VLLM_ASCEND_V41_QNORM_FUSE", "0"/"VLLM_ASCEND_V41_QNORM_FUSE", "1"/' $SRC/vllm_ascend/attention/dsa_v41.py`<br>并在 `scripts/serve.sh` 的 gate 区 export（自文档化） |
| 验证 | ① kernel 计数：`RmsNorm` 140→108/步、`RmsNormDynamicQuant` 3→35/步<br>② `[bneck] hp` p50（单流 ms/step）A/B<br>③ 冒烟 `'2'`；④ 结构化任务逐字一致；⑤ 长文针 60K/74K/150K |
| 回滚 | 把 env 改回 `0` 即回到原路径（patch 本身可保留） |

### 4.2 `0302-slotmap-skip-eager.patch` —— 槽位链死代码消除 【F1/F2】

| 项 | 内容 |
|---|---|
| env | `VLLM_ASCEND_V41_SLOTMAP_SKIP_EAGER`（默认 `0`） |
| 改动点 | `attention/dsa_v41.py::AscendDSAV41MetadataBuilder.build`：把 `compressed_slot_mapping()` 的 eager 调用改成 **仅在"不走 Triton 单发路径"时才执行**；Triton 分支条件提前算成 `use_triton_slots` |
| 为什么是死代码 | Triton kernel 契约明确吃 **raw** slot mapping、自己在 kernel 内套压缩比率（`dsa_v41.py:1099-1105` 注释；`ops/triton/packed_cache_slot_mapping.py`）；`VLLM_ASCEND_V41_TRITON_SLOT_MAPPING` 已被主序列 pin 成 `1`（`envs.py:101-103`、`scripts/serve.sh:263`） |
| 预计收益 | 每次 `build` 省 ≈6-7 个 kernel（`Ge / Add / Remainder / Eq / And / FloorDiv / Where`）。`build` 每步按 KV group 调用（本部署 **13 个 KV group**，`pool_worker.py:288` 日志）；不同 group 之间还会命中 `slot_key` 缓存 ⇒ **保守 0.03~0.13 ms/步 ≈ 0.1~0.3%**（按 6-7 kernel × 1.4 µs × 4~13 次） |
| 可实现性 | Python ✅ |
| 精度风险 | **零**（删的是没人读的计算；若 Triton 路径未被选中则完全走原逻辑） |
| 启用方式 | 同上 pin：`sed -i 's/"VLLM_ASCEND_V41_SLOTMAP_SKIP_EAGER", "0"/..., "1"/'` + `serve.sh` export |
| 验证 | ① 前置确认 `VLLM_ASCEND_V41_TRITON_SLOT_MAPPING=1`（否则本 patch 无效果但不改变行为）<br>② kernel 计数：槽位相关小算子（`SelectV2/FloorDiv/Remainder`）次数下降<br>③ 输出逐字一致（KV 落块正确性是唯一风险点，务必跑长文针） |

### 4.3 `0303-oproj-pad-empty.patch` —— o_proj padding 零填充消除 【F2】

| 项 | 内容 |
|---|---|
| env | `VLLM_ASCEND_V41_OPROJ_PAD_EMPTY`（默认 `0`） |
| 改动点 | `attention/dsa_v41.py::_project_output`：`output.new_zeros(...)` → `output.new_empty(...)`（仅当 attention 结果行数 < 桶行数时触发） |
| 预计收益 | 每层省 1 个 Fill kernel（repo1 profile `Fill 43 次/步 × 1.16 µs ≈ 0.05 ms`）⇒ **≈0.05 ms/步 ≈ 0.12%** |
| 可实现性 | Python ✅ |
| 精度风险 | 零（padding 行是被 runner 丢弃的桶填充行；行间算子无跨行归约） |
| 启用方式 | 同上 pin |
| 验证 | ① `Fill` 计数下降；② 输出逐字一致；③ 覆盖 padding 桶（BAT 图桶 1..32）冒烟 |

---

## 5. 明确不可行 / 不建议（省时间清单）

| 项 | 判定 | 依据 |
|---|---|---|
| 再开 vllm 的 `fuse_norm_quant` / `fuse_act_quant` / `fuse_qknorm_rope` / `fuse_muls_add` | ❌ 无效 | §2：默认已开；vllm-ascend 用自己的 pass manager；pattern 与 V4.1 不匹配；注意力在自定义算子边界内 |
| F4 gating+routing 合并（Python） | ❌ | 两者都已是 CANN 单算子（`fused_topk_router.py:204`、`token_dispatcher.py:492`），合并需 AscendC |
| F6 三件套合并（Python） | ❌ | 两个已是 Triton 单发、一个是 CANN QLI；合成 kernel 需 AscendC |
| F2 最大单项 `output[...] = self.wo_b(...)`（0.256 ms） | ❌ Python | `npu_quant_matmul` 无 `out=`；要么算子 epilogue 直接写目标 buffer，要么取消 `mutates_args` 图边界（影响全模型图捕获，风险高） |
| `hc_pre_norm` / HcPre 侧融合或核数调整 | ❌ 已试尽 | repo1：`hc_pre_norm` E2E +0.36~0.77%、HcPre 核数 34 µs 恒定（`SINGLE-STREAM-LEVERS-EXHAUSTED` §1） |
| AICPU metadata 提前/下沉 | ❌ | 暴露仅 0.03 ms（`FRESH-CONC1...`） |
| `DequantSwigluQuant` 融合 | ❌ | 暴露仅 0.025 ms |
| Eager 模式下的 grouped-topk 链（`grouped_topk_router.py:61-125`） | ❌ 不在活路径 | V4.1 是 `sqrtsoftplus` ⇒ 走 `moe_gating_top_k_hash`（`fused_topk_router.py:204`） |

---

## 6. 优先级清单（按"预计收益 ÷ 实现成本"，≤5 条）

| 序 | 动作 | 预计收益（我们口径） | 成本 | 风险 | 状态 |
|---:|---|---|---|---|---|
| **1** | **`0301` q_norm + wq_b.quantize 融合**（`V41_QNORM_FUSE`） | **0.10~0.15 ms/步（0.25~0.40%）**；repo1 同型实测 −0.62% | 低（已有 patch，1 次重启 A/B） | 中（非逐位，需过精度门） | ✅ patch 就绪 |
| **2** | **`0302` 槽位链死代码消除**（`V41_SLOTMAP_SKIP_EAGER`） | 0.03~0.13 ms/步（0.1~0.3%） | 低 | **零** | ✅ patch 就绪 |
| **3** | **`LNORM` 融合：`input_layernorm` + `wq_a.quantize`**（repo1 的 B 项） | ≈0.13 ms/步（repo1 实测 0.133 ms = −0.52% @24 ms）；它是 `0301` 的姐妹项，35 层可复用同一融合算子 | 中（结构改动：把 norm 结果/量化结果从 `model.py:901` 传进 attention，或写 3 输出 kernel） | 中 | ⏳ 方案（下一步 patch 0304） |
| **4** | **`0303` o_proj padding 零填充消除**（`V41_OPROJ_PAD_EMPTY`） | ≈0.05 ms/步（0.12%） | 低 | **零** | ✅ patch 就绪 |
| **5** | **F4/F6 深度融合（AscendC）立项** | 上界 0.78 ms/步（F4 0.38 + F6 0.40）；现实打折 40~50% ⇒ **0.3~0.4 ms（0.8~1.0%）** | **高**（算子开发 + 精度回归，repo1 经验 ≈2 天/项） | 中 | 📋 仅方案 |

> 前 4 项合计 **≈0.3~0.45 ms/步 ≈ 0.8~1.1%**。要两位数收益只有"多合 1 深度融合"
> （算子数 2821 → <1000），那是算子开发立项，不是 Python 补丁能覆盖的。
> 这与 `extra-info/optimization-report-deepdive.md` §2 的 K1/K6 结论一致。

### 建议顺序与判据

1. 先跑 **0302 + 0303**（零风险，建立 kernel 计数 → ms/step 的折算基线，确认"删算子 = 掉时长"在本部署成立）；
2. 再跑 **0301**（收益最大但要过精度门），同时记录 `RmsNorm`/`RmsNormDynamicQuant` 计数；
3. 三项都过门后，再做 **LNORM 融合（B）**；F4/F6 只在有人力做 AscendC 时启动。

**统一判据（每项都必须过）**：
- 性能：`[bneck] hp` 的单流 ms/step p50（不看聚合 tok/s），同配置两次起服噪声底 ≈0.8%；
- 逐位：结构化任务逐字一致 + 100% 正确（分级门）；
- 长文针：60K/74K/150K 全过；
- kernel 计数差分必须与预期方向一致（否则视为"patch 没生效"，而不是"没收益"）。

---

## 7. 复现/核对入口

```bash
# 源码（本机镜像副本，行号与远端一致）
cd /mnt/sfs/yzl/vllm-ascend-deepseek-v41-a3/rootfs/vllm-workspace/vllm-ascend
sed -n '1058,1195p' vllm_ascend/attention/dsa_v41.py     # F1 槽位链（0302 落点）
sed -n '462,469p'    vllm_ascend/attention/dsa_v41.py     # F2 padding（0303 落点）
sed -n '510,511p'    vllm_ascend/attention/dsa_v41.py     # F3 q_norm/wq_b（0301 落点）
sed -n '1830,1840p'  vllm_ascend/attention/dsa_v1.py      # F2 wo_b 回拷（不可行项的证据）
sed -n '195,340p'    vllm_ascend/models/deepseek_v41/indexer.py   # F6
sed -n '189,270p'    vllm_ascend/ops/fused_moe/router/fused_topk_router.py  # F4 gating
# 上游融合 pass
sed -n '45,80p'  vllm_ascend/compilation/graph_fusion_pass_manager.py
sed -n '35,95p'  vllm_ascend/compilation/passes/norm_quant_fusion_pass.py
# 补丁验证（副本树上，不碰部署树）
mkdir -p /tmp/qf && cp -r ... && git apply --check scripts/patches/experimental/*.patch
```

**纪律确认**：本 task 全程只读远端源码；未启动/重启引擎、未压测、未碰 new_rl3、
未 kill 任何 ssh/frp 进程；写入仅限 `extra-info/fusion-plan.md` 与
`scripts/patches/experimental/{0301,0302,0303}-*.patch`、`series.conf`。
