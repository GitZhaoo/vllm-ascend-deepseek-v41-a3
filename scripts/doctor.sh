#!/usr/bin/env bash
# =============================================================================
# doctor.sh — read-only preflight / health check for this deployment
#
# Checks hardware, rootfs, weights, patch state, ports, disk and configuration.
# It never starts, stops or modifies anything.
#
# Usage:
#   bash scripts/doctor.sh
#   bash scripts/doctor.sh --quiet     # only failures/warnings
# =============================================================================
set -uo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
. "$SELF_DIR/common.sh"

QUIET=0
[ "${1:-}" = "--quiet" ] && QUIET=1

FAIL=0
WARN=0
ok()   { [ "$QUIET" = 1 ] || echo "  ✅ $*"; }
warn() { WARN=$((WARN + 1)); echo "  ⚠  $*"; }
bad()  { FAIL=$((FAIL + 1)); echo "  ❌ $*"; }
section() { [ "$QUIET" = 1 ] || { echo; echo "== $* =="; }; }

dsv41_unset_proxy

section "系统"
case "$(uname -m)" in
  aarch64|arm64) ok "架构 $(uname -m)" ;;
  *) bad "架构 $(uname -m)，需要 aarch64/arm64" ;;
esac
ok "内核 $(uname -r)"

if command -v npu-smi >/dev/null 2>&1; then
  npu_count=$(npu-smi info 2>/dev/null | grep -cE '^\| [0-9]+ +Ascend9')
  if [ "$npu_count" -ge 8 ]; then
    ok "检测到 ${npu_count} 个 Ascend 910 chip 行（8 卡 × 2 die = 16）"
  elif [ "$npu_count" -gt 0 ]; then
    warn "只检测到 ${npu_count} 个 Ascend 910 chip 行，预期 16"
  else
    bad "npu-smi 未报告 Ascend 设备"
  fi
  workers=$(ps -eo comm= 2>/dev/null | grep -c '^VLLM::' || true)
  if [ "${workers:-0}" -gt 0 ]; then
    if [ -n "$(dsv41_find_serve_pid || true)" ]; then
      ok "vLLM 运行中（${workers} 个 VLLM:: 进程）"
    else
      warn "存在 ${workers} 个 VLLM:: 进程但没有对应的 serve 主进程（残留）"
    fi
  else
    ok "无残留 VLLM:: 进程"
  fi
else
  bad "找不到 npu-smi（NPU 驱动未安装或不在 PATH）"
fi

section "目录与磁盘"
for d in "$DSV41_ROOTFS" "$DSV41_RESULTS" "$DSV41_PID_DIR"; do
  if [ -d "$d" ]; then ok "存在 $d"; else
    case "$d" in
      "$DSV41_ROOTFS") bad "rootfs 不存在: $d（运行 scripts/setup_rootfs.sh）" ;;
      *) warn "目录不存在（启动时会创建）: $d" ;;
    esac
  fi
done
free_gib=$(df -Pk "$(dirname "$DSV41_ROOTFS")" 2>/dev/null | awk 'NR==2 {printf "%.1f", $4/1024/1024}')
if [ -n "${free_gib:-}" ]; then
  awk -v v="$free_gib" 'BEGIN {exit !(v < 25)}' && warn "rootfs 所在分区仅剩 ${free_gib} GiB" || ok "rootfs 分区剩余 ${free_gib} GiB"
fi

section "rootfs"
PYHOME=$(find "$DSV41_ROOTFS/usr/local" -maxdepth 1 -type d -name 'python3.12*' 2>/dev/null | sort -V | tail -1)
CANNDIR=$(find "$DSV41_ROOTFS/usr/local/Ascend" -maxdepth 1 -type d -name 'cann-*' 2>/dev/null | sort -V | tail -1)
[ -x "${PYHOME:-/nonexistent}/bin/python3.12" ] && ok "Python: $PYHOME" || bad "rootfs Python 缺失"
[ -f "${CANNDIR:-/nonexistent}/set_env.sh" ] && ok "CANN: $CANNDIR" || bad "rootfs CANN 缺失"
for f in \
  "$DSV41_ROOTFS/vllm-workspace/vllm/vllm/__init__.py" \
  "$DSV41_ROOTFS/vllm-workspace/vllm-ascend/vllm_ascend/envs.py"; do
  [ -f "$f" ] && ok "源码: ${f#"$DSV41_ROOTFS/"}" || bad "源码缺失: $f"
done
if ls "$DSV41_ROOTFS"/vllm-workspace/vllm-ascend/vllm_ascend/vllm_ascend_C*.so >/dev/null 2>&1; then
  ok "预编译扩展 vllm_ascend_C*.so"
else
  warn "未找到 vllm_ascend_C*.so（可能依赖运行时编译）"
fi
[ -f "$DSV41_ROOTFS/.dsv41-image" ] && ok "镜像溯源: $DSV41_ROOTFS/.dsv41-image" \
  || warn "缺少 .dsv41-image（rootfs 可能不是由 scripts/setup_rootfs.sh 生成）"

section "权重"
if [ -f "$DSV41_MODEL/config.json" ]; then
  ok "config.json: $DSV41_MODEL"
  shards=$(find "$DSV41_MODEL" -maxdepth 1 -name 'quant_model_weights-*.safetensors' 2>/dev/null | wc -l)
  if [ "$shards" -eq 142 ]; then ok "权重分片 142 个"; else warn "权重分片 ${shards} 个（参考值 142）"; fi
else
  bad "模型不存在: $DSV41_MODEL/config.json"
fi

section "补丁"
if [ -f "$DSV41_ROOTFS/vllm-workspace/vllm-ascend/vllm_ascend/envs.py" ]; then
  if bash "$SELF_DIR/apply_patches.sh" status 2>/dev/null | grep -q 'verify.*✅'; then
    ok "补丁与 gate 默认值正常"
  else
    warn "补丁状态异常，运行 bash scripts/apply_patches.sh status"
  fi
fi

section "服务"
if [ -n "${DSV41_API_KEY:-}" ]; then
  ok "DSV41_API_KEY 已设置 (len=${#DSV41_API_KEY})"
else
  warn "DSV41_API_KEY 未设置：服务将不带鉴权（仅可用于 loopback 测试）"
fi
pid=$(dsv41_find_serve_pid || true)
if [ -n "$pid" ]; then
  ok "vLLM 运行中 pid=$pid"
  if curl -sf --noproxy '*' --max-time 5 -o /dev/null "http://127.0.0.1:$DSV41_PORT/health"; then
    ok "健康检查 HTTP 200"
  else
    warn "/health 暂不可用（可能正在启动）"
  fi
else
  ok "当前没有运行中的 vLLM（不影响预检）"
fi
listening=0
if command -v ss >/dev/null 2>&1; then
  ss -ltn 2>/dev/null | grep -qE ":${DSV41_PORT}([^0-9]|$)" && listening=1
elif command -v netstat >/dev/null 2>&1; then
  netstat -tln 2>/dev/null | grep -qE ":${DSV41_PORT}([^0-9]|$)" && listening=1
fi
if [ "$listening" = 1 ] && [ -z "$pid" ]; then
  warn "端口 $DSV41_PORT 被其它进程占用"
else
  ok "端口 $DSV41_PORT 可用或由本服务占用"
fi

if [ "${DSV41_LB_MODE:-internal}" = "external" ]; then
  rpid=$(dsv41_find_router_pid || true)
  if [ -n "$rpid" ]; then
    ok "prefix router 运行中 pid=$rpid"
  elif [ -n "$pid" ]; then
    warn "external 模式但 router 未运行（/health 将不可用）"
  fi
  for p in "$DSV41_DP_SUPERVISOR_PORT" $(seq "$DSV41_DP_BASE_PORT" $((DSV41_DP_BASE_PORT + DSV41_DP - 1))); do
    if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -qE ":${p}([^0-9]|$)"; then
      ok "external 端口 $p 监听中"
    else
      ok "external 端口 $p 空闲（服务未运行时正常）"
    fi
  done
fi

section "代理环境"
if env | grep -qiE '^(http|https|all)_proxy='; then
  warn "检测到 HTTP(S)_PROXY；脚本内部会 unset，但手工 curl 请加 --noproxy '*'"
else
  ok "无全局代理变量"
fi

echo
if [ "$FAIL" -gt 0 ]; then
  echo "结果: ❌ ${FAIL} 项失败, ${WARN} 项警告"
  exit 1
fi
if [ "$WARN" -gt 0 ]; then
  echo "结果: ⚠  ${WARN} 项警告（无阻塞项）"
  exit 0
fi
echo "结果: ✅ 全部检查通过"
