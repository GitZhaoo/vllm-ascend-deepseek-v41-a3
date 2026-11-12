# 新机器部署清单（从 git clone 到可服务）

本仓库**只包含代码/脚本/补丁/文档**。以下几样体积过大或与机器相关，**不在 git 里**，
必须在目标机器上单独准备。本文件就是逐项清单与获取方式。

## 1. 不在仓库里的东西

| 项 | 体积 | 为什么不在 git | 获取方式 |
|---|---|---|---|
| **rootfs**（rc2-a3 镜像解包） | ≈21 GB | 二进制镜像，git 无法承载 | 首选 `bash scripts/setup_rootfs.sh`（自动拉 OCI 层，需要能访问 `quay.nju.edu.cn`）；**离线 / 无法访问 registry**: 从健康节点 `tar` 一份，拷到目标机后 `bash scripts/import_rootfs.sh rootfs.tar --dest /cache/dsv41-rc2-rootfs --manifest ref.tsv` |
| **权重**（DeepSeek-V4.1-Flash-1017-w8a8） | ≈728 GB | 同上 | `modelscope download --model Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8 --local_dir <dir>`，或挂载已有共享盘路径 |
| `config/dsv41.local.env` | <10 KB | 机器相关（路径/端口/API key） | `cp config/dsv41.env.example config/dsv41.local.env` 后按目标机器填写 |
| `state/`、`logs/` | — | 运行时产物 | 自动生成 |

> 参考 manifest（用于校验 rootfs 拷贝完整性）：在健康节点上
> `python3 scripts/verify_rootfs.py manifest <rootfs> -o ref.tsv`，把 ref.tsv 拷到新机器使用。
> 实测经历：tar 经共享盘中转曾把符号链接变成 0 字节文件、截断 `.so`，引擎表现为
> `EI0015 HCCL` / `ERR00100 算子加载失败`——**看着像硬件故障，其实是拷贝损坏**。

## 2. 目标机器的前置条件

| 项 | 要求 |
|---|---|
| 硬件 | 单机 8×Ascend 910C（A3，16 die），每 die 64 GB HBM |
| 驱动 | 与镜像配套的 NPU driver（实测 25.2.1 / 25.5.5 均可），`npu-smi info` 能看到 16 die |
| OS / glibc | Ubuntu 22.04 类（glibc ≥2.35）走 native；更旧 glibc（如 Kylin V10）用 `DSV41_RUNTIME=docker` |
| 端口 | 8020（DP supervisor）、8021（router）、8022/8023（rank）、5557/5558（KV events）；KV 池再用 5100/6100/8100 |
| 内存 | ≥1.5 TB 可用（KV 池 16×32 GB = 512 GiB 宿主内存注册） |

## 3. 部署步骤

```bash
git clone <repo> vllm-ascend-deepseek-v41-a3 && cd vllm-ascend-deepseek-v41-a3
cp config/dsv41.env.example config/dsv41.local.env && $EDITOR config/dsv41.local.env  # ROOTFS/MODEL/API_KEY

# rootfs：二选一
bash scripts/setup_rootfs.sh                      # 在线拉取
# bash scripts/import_rootfs.sh rootfs.tar --dest /cache/dsv41-rc2-rootfs --manifest ref.tsv

# 权重
modelscope download --model Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8 --local_dir <dir>

# 补丁（主序列含 0001/0002/0004/0005/0101/0202/0304；实验性 0301-0303/0310/0311 可选）
bash scripts/apply_patches.sh apply && bash scripts/apply_patches.sh verify

# 预检 + 启动
bash scripts/doctor.sh
bash scripts/serve.sh && STATUS=1 bash scripts/serve.sh
python3 tests/smoke_test.py
```

## 4. 启动前必须检查的四个坑（本轮实测踩过）

1. **`/dev/shm` 残留**：KV 池权限下会留下 `memfabric_hybrid_*`（每个 32 GB）；
   如果上一次引擎被 `kill -9`，这些段不会释放，下次启动会在 KV 传输初始化阶段 worker 崩溃。
   → 启动前 `rm -f /dev/shm/memfabric_hybrid_*`（确认没有其它作业在用）。
2. **ZMQ/IPC 残留**：`/tmp/lookup_rpc_port_*`、`/tmp/vllm_rpc_*`；
   残留会让新引擎报 `Address already in use` 或被 router 判成 502。
   → 启动前 `rm -f /tmp/lookup_rpc_port_* /tmp/vllm_rpc_*`。
3. **rootfs 一致性**：见 §1 的 manifest 校验。
4. **NPU 显存未回落**：上一次进程刚退出时显存释放有延迟；
   等到 `npu-smi` 回到 ≈3.2 GB/die 再启动，否则会因 HBM 不足在 KV cache 阶段失败。

## 5. 补丁清单（主序列 vs 实验性）

主序列（`scripts/patches/vllm-ascend-v41-rc2/series.conf`，`apply_patches.sh` 自动应用）：

| 补丁 | 作用 | 默认 |
|---|---|---|
| 0001 / 0002 / 0004 / 0005 | V4.1 优化 gate、DSA metadata、模型算子微优化、packed slot mapping 标量化 | 开 |
| 0101 | KV 池 MRV2 抢占恢复正确性 | 开 |
| 0202 | 多流 vector 核预算（limit_core_num） | 开 |
| 0304 | KV 池 lookup 快路径（跳过冗余 RPC + 合并 13 组查询） | 开 |

实验性（`scripts/patches/experimental/`，需手动应用，env 默认关）：

| 补丁 | 作用 | 状态 |
|---|---|---|
| 0301 / 0302 / 0303 | q_norm+wq_b 融合 / 槽位链死代码跳过 / o_proj 免清零 | 实测 ms/step −0.73%（0301 需精度门） |
| 0310 / 0311 | MegaMoe 排除 draft / 绕过形状白名单 | 备用（V4.1 不支持 MegaMoe，见 docs/optimization-deep-dive） |
