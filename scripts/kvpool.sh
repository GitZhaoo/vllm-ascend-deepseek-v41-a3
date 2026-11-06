#!/usr/bin/env bash
# =============================================================================
# kvpool.sh — external KV pool (AscendStoreConnector + MemCache MetaService)
#
# This follows the reference deployment
# CJYLZS/vllm-ascend-deepseek-v41-a2 (kv_offload.sh):
#
#   * backend = memcache_hybrid (not Mooncake);
#   * transport = host_shm (host_rdma's extend library is a 0-byte stub in
#     this image; device_rdma segfaults at 8+ ranks);
#   * a CPU-only MetaService (no card, no HBM) is started on 5100/6100/8100;
#   * each rank registers DSV41_KV_DRAM_GB of host DRAM (16 dies × 2 GB by
#     default);
#   * the engine is started with
#       --kv-transfer-config '{"kv_connector":"AscendStoreConnector",
#         "kv_role":"kv_both","kv_connector_extra_config":{"backend":"memcache"}}'
#     and MMC_LOCAL_CONFIG_PATH pointing at mmc-local.conf.
#
# The pool is OFF by default. The 0101 patch (MRV2 preemption marker) is a
# correctness gate: without it a preempted request can associate the wrong
# block table with pool keys (silent wrong output).
#
# Usage:
#   bash scripts/kvpool.sh check-capacity   # read-only memory budget check
#   bash scripts/kvpool.sh render-config    # write conf files, print local conf
#   bash scripts/kvpool.sh start            # start/reuse MetaService
#   bash scripts/kvpool.sh stop             # stop the MetaService we started
#   bash scripts/kvpool.sh status           # state + capacity
#   bash scripts/kvpool.sh ensure           # capacity + config + start (serve.sh)
#   bash scripts/kvpool.sh print-env        # effective variables
# =============================================================================
set -Eeuo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
. "$SELF_DIR/common.sh"

RFS=${DSV41_ROOTFS%/}
PY=${DSV41_KV_PYTHON:-}
if [ -z "$PY" ]; then
  PY_HOME=$(find "$RFS/usr/local" -maxdepth 1 -type d -name 'python3.12*' 2>/dev/null | sort -V | tail -1 || true)
  PY="${PY_HOME:-}/bin/python3.12"
fi
CANN=${DSV41_CANN_HOME:-$(find "$RFS/usr/local/Ascend" -maxdepth 1 -type d -name 'cann-*' 2>/dev/null | sort -V | tail -1 || true)}
if [ -n "${CANN:-}" ] && [ -d "$CANN" ]; then
  export LD_LIBRARY_PATH="$CANN/lib64:$CANN/runtime/lib64:${LD_LIBRARY_PATH:-}"
fi

RANKS=$((DSV41_DP * DSV41_TP))
LOCAL_CONF="$DSV41_KVPOOL_STATE_DIR/mmc-local.conf"
META_CONF="$DSV41_KVPOOL_STATE_DIR/mmc-meta.conf"
META_LOG="$DSV41_KV_META_LOG"
META_PID_FILE="$DSV41_KV_META_PID_FILE"
METRICS_URL="http://127.0.0.1:${DSV41_KV_METRICS_PORT}/metrics"

die() { echo "[kvpool] ❌ $*" >&2; exit 1; }
log() { echo "[kvpool] $*" >&2; }
warn() { echo "[kvpool] ⚠ $*" >&2; }

port_listening() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | grep -qE ":${port}([^0-9]|$)"
  else
    netstat -tln 2>/dev/null | grep -qE ":${port}([^0-9]|$)"
  fi
}

metrics_ok() {
  curl -sf --noproxy '*' --max-time 2 "$METRICS_URL" 2>/dev/null \
    | grep -q '^memcache_'
}

meta_pid() {
  [ -f "$META_PID_FILE" ] || return 1
  local pid
  pid=$(cat "$META_PID_FILE" 2>/dev/null || true)
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && echo "$pid"
}

meta_running() {
  meta_pid >/dev/null 2>&1
}

check_capacity() {
  local dram="$DSV41_KV_DRAM_GB" total avail limit suggest
  printf '%s' "$dram" | grep -Eq '^[0-9]+(\.[0-9]+)?$' \
    || die "无效的 DSV41_KV_DRAM_GB=$dram（每 rank 的 GB 数，例如 2）"
  awk -v v="$dram" 'BEGIN {exit !(v > 0)}' \
    || die "DSV41_KV_DRAM_GB=$dram 必须大于 0"

  total=$(awk -v d="$dram" -v r="$RANKS" 'BEGIN {printf "%.1f", d*r}')
  read -r avail_mib total_mib < <(awk '
    /^MemTotal:/ {t=$2}
    /^MemAvailable:/ {a=$2}
    END {printf "%.0f %.0f\n", a/1024, t/1024}
  ' "$DSV41_MEMINFO_FILE" 2>/dev/null || true)
  [ -n "${total_mib:-}" ] && [ "$total_mib" -gt 0 ] \
    || die "无法读取 $DSV41_MEMINFO_FILE 的 MemTotal/MemAvailable"
  avail=$(awk -v m="$avail_mib" 'BEGIN {printf "%.1f", m/1024}')
  limit=$(awk -v t="$total_mib" -v p="$DSV41_KVPOOL_MAX_MEM_PCT" 'BEGIN {printf "%.1f", t/1024*p/100}')
  # Pool must fit both the percentage budget and "keep MIN_FREE_GIB free".
  if awk -v total="$total" -v limit="$limit" -v avail="$avail" \
        -v reserve="$DSV41_KVPOOL_MIN_FREE_GIB" \
        'BEGIN {exit !(total <= limit && total <= avail - reserve)}'; then
    log "容量校验通过：${dram}GB/rank × ${RANKS} = ${total} GiB（MemTotal $(awk -v m="$total_mib" 'BEGIN{printf "%.0f",m/1024}') GiB，可用 ${avail} GiB，上限 ${limit} GiB，保留 ${DSV41_KVPOOL_MIN_FREE_GIB} GiB）"
    return 0
  fi
  suggest=$(awk -v limit="$limit" -v avail="$avail" -v reserve="$DSV41_KVPOOL_MIN_FREE_GIB" -v r="$RANKS" \
    'BEGIN {m=limit; if (avail-reserve < m) m=avail-reserve; v=int(m/r/2)*2; if (v<1) v=1; printf "%d", v}')
  log "容量校验失败：${dram}GB/rank × ${RANKS} = ${total} GiB"
  log "   MemTotal $(awk -v m="$total_mib" 'BEGIN{printf "%.0f",m/1024}') GiB，可用 ${avail} GiB，上限 ${limit} GiB（池 ≤ ${DSV41_KVPOOL_MAX_MEM_PCT}% 且保留 ${DSV41_KVPOOL_MIN_FREE_GIB} GiB）"
  log "   修复：把 DSV41_KV_DRAM_GB 降到 ${suggest}GB（或 DSV41_KV_POOL=0 关闭外部池）后重启。"
  return 1
}

render_config() {
  local dram="$DSV41_KV_DRAM_GB"
  printf '%s' "$dram" | grep -Eq '^[0-9]+(\.[0-9]+)?$' \
    || die "无效的 DSV41_KV_DRAM_GB=$dram"
  case "$DSV41_KV_PROTOCOL" in
    host_shm) ;;
    host_rdma)
      warn "host_rdma 在本镜像不可用：libmf_hybm_accoffload.so 是 0 字节，ExtendLibrary 加载失败"
      ;;
    device_rdma)
      warn "device_rdma 单 rank 能过、8+ rank 必 segfault；默认应使用 host_shm"
      ;;
    *) die "未知 DSV41_KV_PROTOCOL=$DSV41_KV_PROTOCOL（host_shm|host_rdma|device_rdma）" ;;
  esac
  case "$DSV41_KV_META_PORT$DSV41_KV_STORE_PORT$DSV41_KV_METRICS_PORT" in
    *[!0-9]*) die "KV 端口必须是数字（meta/store/metrics）" ;;
  esac
  for p in "$DSV41_KV_META_PORT" "$DSV41_KV_STORE_PORT" "$DSV41_KV_METRICS_PORT"; do
    [ "$p" != "$DSV41_PORT" ] || die "KV 端口 $p 与 vLLM --port 相同"
  done

  mkdir -p "$DSV41_KVPOOL_STATE_DIR"
  cat > "$LOCAL_CONF" <<EOF
ock.mmc.meta_service_url = tcp://127.0.0.1:${DSV41_KV_META_PORT}
ock.mmc.local_service.config_store_url = tcp://127.0.0.1:${DSV41_KV_STORE_PORT}
ock.mmc.log_level = info
ock.mmc.local_service.world_size = ${DSV41_KV_WORLD_SIZE}
ock.mmc.local_service.protocol = ${DSV41_KV_PROTOCOL}
ock.mmc.local_service.dram.size = ${dram}GB
EOF
  cat > "$META_CONF" <<EOF
ock.mmc.meta_service_url = tcp://127.0.0.1:${DSV41_KV_META_PORT}
ock.mmc.meta_service.config_store_url = tcp://127.0.0.1:${DSV41_KV_STORE_PORT}
ock.mmc.meta_service.metrics_url = http://127.0.0.1:${DSV41_KV_METRICS_PORT}
ock.mmc.log_level = info
EOF
  echo "$LOCAL_CONF"
}

require_tree_fix() {
  [ "$DSV41_KV_POOL" = "1" ] || return 0
  local f="$RFS/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py"
  if [ ! -f "$f" ]; then
    if [ "$DSV41_KV_ALLOW_UNFIXED_TREE" = "1" ]; then
      warn "找不到 $f；KV_ALLOW_UNFIXED_TREE=1 放行"
      return 0
    fi
    {
      echo "[kvpool] ❌ KV_OFFLOAD 已开启，但找不到 pool_scheduler.py：$f"
      echo "        无法确认 0101（MRV2 抢占恢复）是否正确应用。"
      echo "        处理：bash scripts/apply_patches.sh apply；或设 DSV41_KV_ALLOW_UNFIXED_TREE=1 明确带病跑。"
    } >&2
    exit 8
  fi
  if grep -q '_preempted_req_ids.discard(request.req_id)' "$f"; then
    return 0
  fi
  if [ "$DSV41_KV_ALLOW_UNFIXED_TREE" = "1" ]; then
    warn "源码树缺 0101（MRV2 抢占恢复）；KV_ALLOW_UNFIXED_TREE=1 放行，可能静默算错"
    return 0
  fi
  {
    echo "[kvpool] ❌ KV_OFFLOAD 已开启，但源码树不含 0101（MRV2 抢占恢复）"
    echo "        后果：抢占后池 key 可能关联到错误的 block，命中池读到别人的 KV，日志不报错。"
    echo "        处理：bash scripts/apply_patches.sh apply（或设 DSV41_KV_ALLOW_UNFIXED_TREE=1 明确带病跑）"
  } >&2
  exit 8
}

start_meta() {
  [ "$DSV41_KV_POOL" = "1" ] || die "DSV41_KV_POOL=0（未启用外部池）"
  require_tree_fix
  [ -x "$PY" ] || die "找不到 rootfs Python: $PY"
  render_config >/dev/null
  mkdir -p "$(dirname "$META_LOG")" "$(dirname "$META_PID_FILE")"

  if meta_running; then
    if metrics_ok; then
      log "MetaService 已在运行 (pid=$(meta_pid), :$DSV41_KV_METRICS_PORT)，复用（池内容保留）"
      return 0
    fi
    warn "MetaService pid=$(meta_pid) 存在但 metrics 不通，重启"
    stop_meta >/dev/null 2>&1 || true
  elif metrics_ok; then
    log "检测到外部 MetaService（:$DSV41_KV_METRICS_PORT），复用"
    return 0
  fi

  local p miss=""
  for p in "$DSV41_KV_META_PORT" "$DSV41_KV_STORE_PORT" "$DSV41_KV_METRICS_PORT"; do
    port_listening "$p" && miss="$miss $p"
  done
  if [ -n "$miss" ]; then
    die "启动 MetaService 前端口已被占用:$miss（查 ss -ltnp；或复用/停止旧实例）"
  fi

  log "启动 MetaService → :$DSV41_KV_META_PORT/:$DSV41_KV_STORE_PORT/:$DSV41_KV_METRICS_PORT，日志 $META_LOG"
  MMC_META_CONFIG_PATH="$META_CONF" \
    nohup "$PY" -c 'from memcache_hybrid import MetaService; MetaService.main()' \
      >"$META_LOG" 2>&1 &
  local pid=$!
  echo "$pid" > "$META_PID_FILE"
  local i
  for i in $(seq 1 60); do
    if metrics_ok; then
      log "✅ MetaService 就绪 (pid=$pid)"
      return 0
    fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  echo "[kvpool] ❌ MetaService 60s 内未就绪" >&2
  tail -n 30 "$META_LOG" >&2 || true
  kill "$pid" 2>/dev/null || true
  rm -f "$META_PID_FILE"
  exit 6
}

stop_meta() {
  local pid
  pid=$(meta_pid || true)
  if [ -n "$pid" ]; then
    log "停止 MetaService pid=$pid"
    kill "$pid" 2>/dev/null || true
    local i
    for i in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$META_PID_FILE"
  if metrics_ok; then
    warn ":$DSV41_KV_METRICS_PORT/metrics 仍有响应（可能是外部 MetaService，未停止）"
  fi
}

show_status() {
  echo "[kvpool] kv_pool=$DSV41_KV_POOL backend=$DSV41_KV_BACKEND protocol=$DSV41_KV_PROTOCOL dram=${DSV41_KV_DRAM_GB}GB/rank ranks=$RANKS"
  echo "[kvpool] meta=:$DSV41_KV_META_PORT store=:$DSV41_KV_STORE_PORT metrics=:$DSV41_KV_METRICS_PORT conf=$LOCAL_CONF"
  if pid=$(meta_pid 2>/dev/null); then
    echo "[kvpool] MetaService 运行中 pid=$pid"
  elif metrics_ok; then
    echo "[kvpool] MetaService 由外部启动（metrics 可达）"
  else
    echo "[kvpool] MetaService 未运行"
  fi
  if metrics_ok; then
    curl -s --noproxy '*' --max-time 3 "$METRICS_URL" \
      | awk '/^memcache_total_capacity_bytes/ {printf "[kvpool] %s %s\n", $1, $2}'
  fi
}

print_env() {
  cat <<EOF
DSV41_KV_POOL=$DSV41_KV_POOL
DSV41_KV_DRAM_GB=$DSV41_KV_DRAM_GB
DSV41_KV_PROTOCOL=$DSV41_KV_PROTOCOL
DSV41_KV_WORLD_SIZE=$DSV41_KV_WORLD_SIZE
DSV41_KV_META_PORT=$DSV41_KV_META_PORT
DSV41_KV_STORE_PORT=$DSV41_KV_STORE_PORT
DSV41_KV_METRICS_PORT=$DSV41_KV_METRICS_PORT
DSV41_KV_BACKEND=$DSV41_KV_BACKEND
DSV41_KV_META_LOG=$DSV41_KV_META_LOG
DSV41_KV_META_PID_FILE=$DSV41_KV_META_PID_FILE
DSV41_KV_ALLOW_UNFIXED_TREE=$DSV41_KV_ALLOW_UNFIXED_TREE
DSV41_KVPOOL_STATE_DIR=$DSV41_KVPOOL_STATE_DIR
DSV41_KVPOOL_MIN_FREE_GIB=$DSV41_KVPOOL_MIN_FREE_GIB
DSV41_KVPOOL_MAX_MEM_PCT=$DSV41_KVPOOL_MAX_MEM_PCT
EOF
}

case "${1:-status}" in
  check-capacity) check_capacity ;;
  render-config)  render_config ;;
  start)          start_meta ;;
  stop)           stop_meta ;;
  status)         show_status ;;
  print-env)      print_env ;;
  ensure)
    [ "$DSV41_KV_POOL" = "1" ] || die "DSV41_KV_POOL=0（未启用外部池）"
    check_capacity
    require_tree_fix
    render_config >/dev/null
    start_meta
    log "✅ kv pool 就绪 config=$LOCAL_CONF"
    echo "$LOCAL_CONF"
    ;;
  *) echo "用法: $0 {check-capacity|render-config|start|stop|status|ensure|print-env}"; exit 1 ;;
esac
