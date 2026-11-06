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
#   bash scripts/doctor.sh --probe     # also import torch_npu/vllm_ascend with the
#                                      # serving environment (needs no running server)
#
# With DSV41_RUNTIME=docker the rootfs checks are replaced by checks of the
# docker daemon, the image, the container and the sources inside it.
# =============================================================================
set -uo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
. "$SELF_DIR/common.sh"

QUIET=0
PROBE=0
for _arg in "$@"; do
  case "$_arg" in
    --quiet) QUIET=1 ;;
    --probe) PROBE=1 ;;
  esac
done
unset _arg

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
  mine=$(dsv41_instance_pids | wc -l | tr -d ' ')
  if [ -n "$(dsv41_find_serve_pid || true)" ]; then
    ok "本部署 vLLM 运行中（带标记进程 ${mine} 个；本机共 ${workers:-0} 个 VLLM:: 进程）"
  elif [ "${mine:-0}" -gt 0 ]; then
    warn "本部署有 ${mine} 个残留进程但没有 serve 主进程（bash scripts/kill_vllm.sh 只会清理这些）"
  elif [ "${workers:-0}" -gt 0 ]; then
    ok "本机有 ${workers} 个其它 VLLM:: 进程（非本部署；本项目脚本不会清理它们）"
  else
    ok "无残留 VLLM:: 进程"
  fi
else
  bad "找不到 npu-smi（NPU 驱动未安装或不在 PATH）"
fi

section "目录与磁盘"
if [ "$DSV41_RUNTIME" = "docker" ]; then
  check_dirs="$DSV41_RESULTS $DSV41_PID_DIR"
  disk_ref="$DSV41_PROJECT_ROOT"
  disk_min=10
  case "$DSV41_PID_DIR" in
    /|/tmp) bad "docker 运行时 DSV41_PID_DIR 不能是 '$DSV41_PID_DIR'，请改为专用目录（如 /run/dsv41）" ;;
  esac
else
  check_dirs="$DSV41_ROOTFS $DSV41_RESULTS $DSV41_PID_DIR"
  disk_ref="$(dirname "$DSV41_ROOTFS")"
  disk_min=25
fi
for d in $check_dirs; do
  if [ -d "$d" ]; then ok "存在 $d"; else
    case "$d" in
      "$DSV41_ROOTFS") bad "rootfs 不存在: $d（运行 scripts/setup_rootfs.sh）" ;;
      *) warn "目录不存在（启动时会创建）: $d" ;;
    esac
  fi
done
free_gib=$(df -Pk "$disk_ref" 2>/dev/null | awk 'NR==2 {printf "%.1f", $4/1024/1024}')
if [ -n "${free_gib:-}" ]; then
  awk -v v="$free_gib" -v m="$disk_min" 'BEGIN {exit !(v < m)}' \
    && warn "$disk_ref 所在分区仅剩 ${free_gib} GiB" || ok "$disk_ref 所在分区剩余 ${free_gib} GiB"
fi

CONTAINER_RUNNING=0
if [ "$DSV41_RUNTIME" = "docker" ]; then
  section "容器运行时 (docker)"
  if ! command -v docker >/dev/null 2>&1; then
    bad "找不到 docker 命令"
  elif ! docker info >/dev/null 2>&1; then
    bad "docker daemon 不可用（systemctl status docker）"
  else
    ok "docker $(docker version --format '{{.Server.Version}}' 2>/dev/null)"
    if docker image inspect "$DSV41_IMAGE" >/dev/null 2>&1; then
      ok "镜像已加载: $DSV41_IMAGE"
    else
      bad "本机没有镜像 $DSV41_IMAGE（离线环境: docker load -i <镜像.tar>）"
    fi
    cstate=$(docker inspect -f '{{.State.Status}}' "$DSV41_CONTAINER" 2>/dev/null || true)
    if [ "$cstate" = "running" ]; then
      CONTAINER_RUNNING=1
      ok "容器 $DSV41_CONTAINER 运行中"
      dx() { docker exec "$DSV41_CONTAINER" "$@"; }
      ok "glibc: 宿主 [$(ldd --version 2>&1 | head -1)] / 容器 [$(dx bash -c 'ldd --version 2>&1 | head -1')]"
      ct_py=$(dx bash -c 'ls -d /usr/local/python3.12* 2>/dev/null | sort -V | tail -1')
      ct_cann=$(dx bash -c 'ls -d /usr/local/Ascend/cann-* 2>/dev/null | sort -V | tail -1')
      [ -n "$ct_py" ] && ok "容器 Python: $ct_py" || bad "容器内找不到 /usr/local/python3.12*"
      [ -n "$ct_cann" ] && ok "容器 CANN: $ct_cann" || bad "容器内找不到 /usr/local/Ascend/cann-*"
      dx test -f /vllm-workspace/vllm/vllm/__init__.py && ok "源码: vllm-workspace/vllm" || bad "容器内缺 vllm 源码"
      dx test -f /vllm-workspace/vllm-ascend/vllm_ascend/envs.py && ok "源码: vllm-workspace/vllm-ascend" || bad "容器内缺 vllm-ascend 源码"
      if dx bash -c 'ls /vllm-workspace/vllm-ascend/vllm_ascend/vllm_ascend_C*.so >/dev/null 2>&1'; then
        ok "预编译扩展 vllm_ascend_C*.so"
      else
        warn "未找到 vllm_ascend_C*.so（可能依赖运行时编译）"
      fi
      dx test -d /usr/local/Ascend/driver/lib64 && ok "容器内可见宿主驱动 lib64" || bad "容器内看不到 /usr/local/Ascend/driver/lib64（挂载缺失）"
      dx test -x /usr/local/bin/npu-smi && ok "容器内 npu-smi 可用" || warn "容器内没有 npu-smi"
      dx test -f "$DSV41_MODEL/config.json" && ok "容器内可见模型: $DSV41_MODEL" || bad "容器内看不到模型 $DSV41_MODEL（需要同路径挂载）"
    else
      bad "容器 $DSV41_CONTAINER 未运行（状态: ${cstate:-不存在}）：bash scripts/container.sh up"
    fi
  fi
else
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
fi

if [ "$PROBE" = 1 ]; then
  section "环境探针 (torch_npu / vllm_ascend)"
  if [ "$DSV41_RUNTIME" = "docker" ] && [ "$CONTAINER_RUNNING" != 1 ]; then
    warn "容器未运行，跳过探针"
  else
    probe_out=$(DSV41_PROBE=1 bash "$SELF_DIR/serve.sh" 2>&1)
    probe_rc=$?
    want_npus=$((DSV41_DP * DSV41_TP))
    npu_seen=$(printf '%s\n' "$probe_out" | awk '/^npu_count/ {print $2}' | tail -1)
    if [ "$probe_rc" -ne 0 ]; then
      bad "探针失败（rc=$probe_rc）："
      printf '%s\n' "$probe_out" | tail -n 15 | sed 's/^/      /'
    elif [ "${npu_seen:-0}" -ge "$want_npus" ]; then
      ok "探针通过：$(printf '%s\n' "$probe_out" | tr '\n' ' ')"
    else
      bad "探针只看到 ${npu_seen:-0} 个 NPU，DP×TP 需要 ${want_npus}"
    fi
  fi
fi

section "权重"
if [ -f "$DSV41_MODEL/config.json" ]; then
  ok "config.json: $DSV41_MODEL"
  shards=$(find "$DSV41_MODEL" -maxdepth 1 -name 'quant_model_weights-*.safetensors' 2>/dev/null | wc -l)
  if [ "$shards" -eq 142 ]; then ok "权重分片 142 个"; else warn "权重分片 ${shards} 个（参考值 142）"; fi
else
  bad "模型不存在: $DSV41_MODEL/config.json"
fi

section "补丁"
if { [ "$DSV41_RUNTIME" = "docker" ] && [ "$CONTAINER_RUNNING" = 1 ]; } \
   || { [ "$DSV41_RUNTIME" != "docker" ] && [ -f "$DSV41_ROOTFS/vllm-workspace/vllm-ascend/vllm_ascend/envs.py" ]; }; then
  patch_status=$(bash "$SELF_DIR/apply_patches.sh" status 2>/dev/null)
  patch_rc=$?
  if [ "$patch_rc" -eq 0 ] && grep -q 'verify.*✅' <<<"$patch_status"; then
    ok "补丁与 gate 默认值正常"
  else
    warn "补丁状态异常，运行 bash scripts/apply_patches.sh status"
  fi
else
  warn "跳过补丁检查（rootfs / 容器不可用）"
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

section "运行期开关"
if [ -n "$pid" ]; then
  want_npus=$((DSV41_DP * DSV41_TP))

  # Worker / engine environment: read from the running processes, not the
  # doctor shell (serve.sh exports these into the engine's environment).
  hccl_det=""
  hccl_aiv=""
  for p in $(dsv41_instance_pids 2>/dev/null); do
    [ -r "/proc/$p/environ" ] || continue
    v=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null)
    [ -z "$hccl_det" ] && hccl_det=$(printf '%s\n' "$v" | sed -n 's/^HCCL_DETERMINISTIC=//p' | head -n1)
    [ -z "$hccl_aiv" ] && hccl_aiv=$(printf '%s\n' "$v" | sed -n 's/^HCCL_OP_EXPANSION_MODE=//p' | head -n1)
  done

  if [ "$DSV41_RUNTIME" = "docker" ]; then
    # plog/serve logs live in the state dir mounted into the container.
    plog_root="${DSV41_STATE_DIR}/ascend/log/run/plog"
    v2_src=""
    if command -v "$DSV41_DOCKER_BIN" >/dev/null 2>&1 && "$DSV41_DOCKER_BIN" inspect -f '{{.State.Running}}' "$DSV41_CONTAINER" >/dev/null 2>&1; then
      v2_src="$("$DSV41_DOCKER_BIN" logs "$DSV41_CONTAINER" 2>/dev/null)"
    fi
  else
    plog_root="${DSV41_PLOG_DIR:-/root/ascend/log/run/plog}"
    v2_src=""
    latest_serve=$(ls -t "$DSV41_RESULTS"/serve_*.log 2>/dev/null | head -n 1 || true)
    [ -n "$latest_serve" ] && v2_src=$(cat "$latest_serve" 2>/dev/null)
  fi
  if [ -z "$v2_src" ]; then
    warn "找不到 serve 日志，无法确认 V2 model runner"
  else
    v2_ranks=$(printf '%s\n' "$v2_src" | grep -c 'npu model runner v2 is in developing')
    if [ "${v2_ranks:-0}" -ge "$want_npus" ]; then
      ok "V2 model runner 已生效（${v2_ranks}/${want_npus} rank）"
    else
      warn "V2 model runner 日志只看到 ${v2_ranks:-0}/${want_npus} rank（V1 判据在 V2 下恒为 0，不要用旧判据）"
    fi
  fi

  # AIV: count ranks whose plog emitted aivMode[1].
  aiv_ranks=0
  if [ -d "$plog_root" ]; then
    for p in $(dsv41_instance_pids 2>/dev/null); do
      f=$(find "$plog_root" -maxdepth 1 -type f -name "plog-${p}_*" -print -quit 2>/dev/null || true)
      if [ -n "$f" ] && grep -q 'aivMode\[1\]' "$f" 2>/dev/null; then
        aiv_ranks=$((aiv_ranks + 1))
      fi
    done
    if [ "$aiv_ranks" -ge "$want_npus" ]; then
      ok "HCCL AIV 已生效（plog aivMode[1]：${aiv_ranks}/${want_npus} rank）"
    else
      bad "HCCL AIV 未确认（plog aivMode[1]：${aiv_ranks}/${want_npus} rank；检查 HCCL_OP_EXPANSION_MODE / HCCL_DETERMINISTIC）"
    fi
  else
    warn "找不到 plog 目录 $plog_root，无法确认 AIV（HCCL_OP_EXPANSION_MODE=$hccl_aiv）"
  fi

  case "${hccl_det:-}" in
    1|true|TRUE|on|ON) warn "HCCL_DETERMINISTIC=$hccl_det 会静默压掉 AIV 展开，确认是否为压测临时设置" ;;
  esac
  case "${hccl_aiv:-AIV}" in
    AIV) ;;
    *) warn "HCCL_OP_EXPANSION_MODE=$hccl_aiv（非 AIV）；serve.sh 默认传 AIV，确认是否被覆盖" ;;
  esac
else
  ok "服务未运行，跳过运行期开关自证"
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
