# 实验性补丁（未进入生产补丁序列）

## 0202-limit-core-num.patch

来源：CANN 低时延 recipe（DeepSeek-V4.1 indexer）的「多流 + 控核」范式，仅 A3 可用。
作用：给 `dsa_v41.py` 的两条重叠流分配 vector 核预算 —— 主流是 cube 密集（wq_a/wq_b
matmul，不吃 vector 核），重叠流是 vector 密集（quantize / kv_norm / RoPE / SWA 写）。
默认**完全关闭**（env 未设置时行为与官方一致）：

```bash
VLLM_ASCEND_V41_LIMIT_CORE_MAIN_AIV=16   # 主流保留 16/48 vector 核
VLLM_ASCEND_V41_LIMIT_CORE_AUX_AIV=32    # 重叠流 32/48 vector 核
VLLM_ASCEND_V41_LIMIT_CORE_AIC=24        # 可选，默认 24（不缩 cube 预算）
```

应用于 rootfs 副本：
```bash
cd <rootfs>/vllm-workspace/vllm-ascend && patch -p1 < .../0202-limit-core-num.patch
```
