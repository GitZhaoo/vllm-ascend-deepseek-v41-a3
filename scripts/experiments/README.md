# A/B 实验工具链（2026-11-11 单流/TTFT 评估）

在**独立测试节点**上复制线上的运行时（同一 rootfs 拷贝 + 相同启动参数），
用环境变量覆盖做单变量 A/B，用于评估「单流 tok/s / TTFT / 双前缀并发」。

| 文件 | 作用 |
|---|---|
| `bench_suite.py` | 固定 5 组口径：单流热前缀 / 短prompt TTFT / 冷前缀 TTFT / 双前缀 c2,16,64 / `/metrics` 拆分（ms/step、接受率、引擎 prefill） |
| `run_config.sh` | 起→等就绪→压测→取证→停；HOME 隔离编译缓存；清理按 `DSV41_PROC_TAG` |
| `envs/*.env` | 单变量 overlay（graph 模式 / MC2 / kernel / max-seqs / BAT / KV 池） |
| `patch_serve.py` | 只给**测试副本**的 serve.sh 注入 knob，生产脚本不受影响 |
| `summarize.py` | 把 JSON 汇总成 markdown 对比表 |

前置：`DSV41_ROOTFS` 指向一份 rootfs 副本，`DSV41_MODEL` 指向权重，
`EXP=/cache/exp`（实验目录），并把仓库拷贝到 `$EXP/repo-test`。

结果（2026-11-11）：
- 基线复现：单流 102.5 tok/s、TTFT 303 ms、双前缀 c16/c64 = 1333/1993；
- `FULL_AND_PIECEWISE` 无 TTFT 收益且并发回退；`fused_mc2` 输出退化；
  `static_kernel(+npugraph_ex)` ERR00100 起不来；`prefill_mc2`@BAT16384 OOM。
- 详见 [`docs/optimization-2026-11-11.md`](../../docs/optimization-2026-11-11.md)。
