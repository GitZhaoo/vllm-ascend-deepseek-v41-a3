# MegaMoe 实验补丁（默认全部关闭）

> 配套分析：`extra-info/megamoe-analysis.md`
> 注意：本目录同时被其他实验任务使用；本文件只描述下面两个 MegaMoe 补丁，
> `series.conf` 由别的任务维护，**不要**直接覆盖。

两个补丁都**不自动应用**，也不在 `series.conf` 里登记；需要时手工 `git apply`
或拷贝到 `repo-test` 后按实验流程启动。

## 0310-megamoe-exclude-draft.patch

env：`VLLM_ASCEND_V41_MEGAMOE_EXCLUDE_DRAFT=1`

让 DSpark/MTP draft（3 个 draft block，128 routed experts，topk 3）在
`enable_fused_mc2=2`（MegaMoe 生效）时**不**走 FUSED_MC2，而是回落到普通
MC2/ALLTOALL；target 模型仍按原有策略走 MegaMoe。

- 基线：本补丁的 diff baseline 是**已经打完 Lead 的 prefill-only 门槛补丁**
  （`VLLM_ASCEND_V41_MEGAMOE_PREFILL_ONLY` / `_MIN_TOKENS`，远端
  `ascend_forward_context.py` 当前状态）的树；如果从干净树打，先打 prefill-only
  补丁再打本补丁。
- 权重安全性：W8A8 MegaMoe 权重准备（`w8a8_dynamic.py:process_weights_after_loading`
  的 `elif use_cann_megamoe(...)` 分支）只**新增** `cann_mega_moe_*` / 每 expert
  视图，`layer.w13_weight`、`layer.w2_weight` 仍然保留，因此降到 MC2 后
  `_get_mlp_weights()` / `apply_moe_mlp` 路径可用。
- 已知局限：draft 的 MegaMoe symm buffer 与 target 用不同的
  `(MoECommType, num_experts, num_local_experts)` key（128 vs 384 experts），
  一旦 draft 触发过一次 MegaMoe，它那份 symm buffer 已经分配；本补丁只是不再
  选择 FUSED_MC2，不会回收那块通信内存（进程生命周期内固定）。

## 0311-megamoe-allow-unsupported-shape.patch

env：`VLLM_ASCEND_V41_MEGAMOE_ALLOW_UNSUPPORTED_SHAPE=1`

绕过 `AscendConfig._is_megamoe_supported_by_config()` 的 A3 形状白名单，
让 `additional_config.enable_fused_mc2=2` 真正生效（否则会被静默回滚成 0，
见分析文档第 1 节）。**仅在确认 CANN MegaMoe 算子能 tile 当前形状时使用**，
否则会在启动/profile 阶段报 tiling 错误或精度异常。

- baseline：`ascend_config.py` 干净文件（该文件当前未被其它补丁修改）。
- 打开后 `multistream_overlap_shared_expert` 仍会被互斥逻辑强制置 False
  （`ascend_config.py:842`），这是已知的连带代价。

## 环境变量生效方式（重要）

本仓库约定（见 `series.conf`）：spawn 出来的 worker 可能**不继承**
`os.environ`；因此实际启用时优先把补丁里的默认值直接 pin 成 `"1"`
（就像主序列 `0202` 的做法），或在 `scripts/serve.sh` 在同一进程里 export 后再
spawn，并通过启动日志里的 warning / `sym-buffer alloc` 日志确认真的生效，
不要只看 env 是否设置。

## 应用 / 回滚示例

```bash
SRC=/cache/dsv41-rc2-rootfs/vllm-workspace/vllm-ascend \
  PATCH_DIR=$PWD/scripts/patches/experimental \
  bash scripts/apply_patches.sh apply     # 只应包含你要的补丁，注意与别的任务共用目录
```
