#!/usr/bin/env bash
# =============================================================================
# kvpool.sh — Mooncake KV pool lifecycle (AscendStoreConnector)
#
# The external KV pool lets vLLM keep prefix-cache blocks in host memory
# (registered through Mooncake) instead of only in NPU HBM, so repeated
# prefixes — long documents, multi-turn chat, shared system prompts — hit the
# cache across requests and across DP ranks instead of being recomputed.
#
# serve.sh calls `kvpool.sh ensure` before launching the engine when
# DSV41_KV_POOL=1 (the default). `ensure` writes the Mooncake config and starts
# mooncake_master unless something already listens on the master port.
#
# Commands:
#   ensure         write config + start master if needed (used by serve.sh)
#   start          start mooncake_master (idempotent; reuses an external one)
#   stop           stop the harness-started master only
#   status         show pid / port / configured segment size
#   render-config  write mooncake.json and print its path
#   print-env      print the effective settings (debugging)
#
# Rollback: DSV41_KV_POOL=0 disables the pool entirely (serve.sh then starts
# the engine exactly as before). Config lives in config/dsv41.local.env.
# =============================================================================
set -Eeuo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
. "$SELF_DIR/common.sh"

die() { echo "[kvpool] ❌ $*" >&2; exit 1; }
log() { echo "[kvpool] $*" >&2; }

MASTER_HOST=${DSV41_MOONCAKE_MASTER%:*}
MASTER_PORT=${DSV41_MOONCAKE_MASTER##*:}
MOONCAKE_CFG="$DSV41_KVPOOL_STATE_DIR/mooncake.json"
RFS=${DSV41_ROOTFS%/}

# /dev/tcp probe keeps this dependency-free on hosts without ss/netcat.
# A 2 s timeout avoids hanging on an unreachable remote master address.
port_open() {
  if command -v timeout >/dev/null 2>&1; then
    timeout 2 bash -c "exec 3<>/dev/tcp/$1/$2" 2>/dev/null
  else
    (exec 3<>"/dev/tcp/$1/$2") 2>/dev/null
  fi
}

rootfs_python() {
  local home=${DSV41_PYTHONHOME:-$(find "$RFS/usr/local" -maxdepth 1 -type d -name 'python3.12*' 2>/dev/null | sort -V | tail -1)}
  if [ -z "$home" ] || [ ! -x "$home/bin/python3.12" ]; then
    die "找不到 rootfs Python（先 bash scripts/setup_rootfs.sh，或设置 DSV41_PYTHONHOME）"
  fi
  echo "$home/bin/python3.12"
}

render_config() {
  mkdir -p "$DSV41_KVPOOL_STATE_DIR"
  local ssd_flag=false ssd_path=""
  if [ "$DSV41_MOONCAKE_SSD_OFFLOAD" = "1" ]; then
    [ -n "$DSV41_MOONCAKE_SSD_PATH" ] || die "DSV41_MOONCAKE_SSD_OFFLOAD=1 需要 DSV41_MOONCAKE_SSD_PATH"
    ssd_flag=true
    ssd_path=$DSV41_MOONCAKE_SSD_PATH
  fi
  cat > "$MOONCAKE_CFG" <<JSON
{
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "ascend",
  "device_name": "",
  "master_server_address": "$DSV41_MOONCAKE_MASTER",
  "global_segment_size": "$DSV41_MOONCAKE_SEGMENT_SIZE",
  "local_buffer_size": "$DSV41_MOONCAKE_LOCAL_BUFFER",
  "preferred_segment": false,
  "prefer_alloc_in_same_node": true,
  "enable_ssd_offload": $ssd_flag,
  "ssd_offload_path": "$ssd_path",
  "tenant_id": "$DSV41_MOONCAKE_TENANT"
}
JSON
  echo "$MOONCAKE_CFG"
}

master_pid() {
  local pid=""
  [ -f "$DSV41_MOONCAKE_PID_FILE" ] && pid=$(cat "$DSV41_MOONCAKE_PID_FILE" 2>/dev/null || true)
  # Guard against PID reuse after a reboot: only trust the pidfile when the
  # process really is mooncake_master.
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null \
    && tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "mooncake_master"; then
    echo "$pid"
    return 0
  fi
  return 1
}

cmd_start() {
  local pid
  if pid=$(master_pid); then
    log "mooncake_master 已在运行 (pid=$pid, $DSV41_MOONCAKE_MASTER)"
    return 0
  fi
  if port_open "$MASTER_HOST" "$MASTER_PORT"; then
    log "端口 $DSV41_MOONCAKE_MASTER 已有 master 监听（复用外部 master）"
    return 0
  fi

  local py bin home launcher
  py=$(rootfs_python)
  home=${py%/bin/python3.12}
  bin=${DSV41_MOONCAKE_MASTER_BIN:-$home/bin/mooncake_master}
  [ -x "$bin" ] || die "找不到 mooncake_master: $bin（设置 DSV41_MOONCAKE_MASTER_BIN）"
  # setsid puts the entry script and the real binary it spawns in one process
  # group, so stop can terminate the whole tree.
  launcher=setsid
  command -v setsid >/dev/null 2>&1 || launcher=nohup

  render_config >/dev/null
  mkdir -p "$(dirname "$DSV41_MOONCAKE_LOG")"
  log "启动 mooncake_master → $DSV41_MOONCAKE_MASTER (metrics :$DSV41_MOONCAKE_METRICS_PORT，日志 $DSV41_MOONCAKE_LOG)"
  HOME="${HOME:-/root}" PYTHONHOME="$home" \
    PYTHONPATH="$RFS/vllm-workspace/vllm-ascend:$RFS/vllm-workspace/vllm" \
    "$launcher" "$bin" "--rpc_port=$MASTER_PORT" "--metrics_port=$DSV41_MOONCAKE_METRICS_PORT" \
      >"$DSV41_MOONCAKE_LOG" 2>&1 &
  pid=$!
  echo "$pid" > "$DSV41_MOONCAKE_PID_FILE"

  local i
  for i in $(seq 1 30); do
    port_open "$MASTER_HOST" "$MASTER_PORT" && break
    kill -0 "$pid" 2>/dev/null || { tail -n 20 "$DSV41_MOONCAKE_LOG" >&2 || true; die "mooncake_master 启动失败"; }
    sleep 1
  done
  port_open "$MASTER_HOST" "$MASTER_PORT" || { tail -n 20 "$DSV41_MOONCAKE_LOG" >&2 || true; die "mooncake_master 30s 内未监听 $DSV41_MOONCAKE_MASTER"; }
  log "mooncake_master 就绪 (pid=$pid)"
}

cmd_stop() {
  local pid pgid own_pgid
  if ! pid=$(master_pid); then
    log "没有 harness 启动的 mooncake_master 在运行"
    rm -f "$DSV41_MOONCAKE_PID_FILE"
    return 0
  fi
  # The mooncake_master entry script spawns the real binary as a child; it is
  # started with setsid so the whole tree shares one process group.
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)
  own_pgid=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ' || true)
  log "停止 mooncake_master (pid=$pid${pgid:+, pgid=$pgid})"
  if [ -n "$pgid" ] && [ "$pgid" != "$own_pgid" ]; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
  else
    kill -TERM "$pid" 2>/dev/null || true
  fi
  local i
  for i in $(seq 1 15); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    if [ -n "$pgid" ] && [ "$pgid" != "$own_pgid" ]; then
      kill -KILL -- "-$pgid" 2>/dev/null || true
    else
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$DSV41_MOONCAKE_PID_FILE"
  if port_open "$MASTER_HOST" "$MASTER_PORT"; then
    log "⚠️ $DSV41_MOONCAKE_MASTER 仍有监听（可能是外部 master，未被停止）"
  else
    log "mooncake_master 已停止"
  fi
}

cmd_status() {
  local pid
  if pid=$(master_pid); then
    echo "[kvpool] master 运行中 pid=$pid addr=$DSV41_MOONCAKE_MASTER metrics=:$DSV41_MOONCAKE_METRICS_PORT"
  elif port_open "$MASTER_HOST" "$MASTER_PORT"; then
    echo "[kvpool] 端口 $DSV41_MOONCAKE_MASTER 有外部 master 监听（非本脚本启动）"
  else
    echo "[kvpool] master 未运行"
    return 1
  fi
  echo "[kvpool] kv_pool=$DSV41_KV_POOL backend=$DSV41_MOONCAKE_BACKEND segment=${DSV41_MOONCAKE_SEGMENT_SIZE}/card tenant=$DSV41_MOONCAKE_TENANT"
  echo "[kvpool] config=$MOONCAKE_CFG"
}

cmd_ensure() {
  [ "$DSV41_KV_POOL" = "1" ] || die "DSV41_KV_POOL=0（ensure 只在启用时调用）"
  render_config >/dev/null
  cmd_start
  log "✅ kv pool 就绪 config=$MOONCAKE_CFG master=$DSV41_MOONCAKE_MASTER segment=${DSV41_MOONCAKE_SEGMENT_SIZE}/card"
  # stdout is consumed by serve.sh: path to mooncake.json.
  echo "$MOONCAKE_CFG"
}

cmd_print_env() {
  cat <<ENV
DSV41_KV_POOL=$DSV41_KV_POOL
DSV41_MOONCAKE_MASTER=$DSV41_MOONCAKE_MASTER
DSV41_MOONCAKE_METRICS_PORT=$DSV41_MOONCAKE_METRICS_PORT
DSV41_MOONCAKE_SEGMENT_SIZE=$DSV41_MOONCAKE_SEGMENT_SIZE
DSV41_MOONCAKE_LOCAL_BUFFER=$DSV41_MOONCAKE_LOCAL_BUFFER
DSV41_MOONCAKE_BACKEND=$DSV41_MOONCAKE_BACKEND
DSV41_MOONCAKE_TENANT=$DSV41_MOONCAKE_TENANT
DSV41_MOONCAKE_SSD_OFFLOAD=$DSV41_MOONCAKE_SSD_OFFLOAD
DSV41_MOONCAKE_SSD_PATH=$DSV41_MOONCAKE_SSD_PATH
DSV41_KVPOOL_STATE_DIR=$DSV41_KVPOOL_STATE_DIR
DSV41_MOONCAKE_CONFIG=$MOONCAKE_CFG
ENV
}

case "${1:-}" in
  start)         cmd_start ;;
  stop)          cmd_stop ;;
  status)        cmd_status ;;
  ensure)        cmd_ensure ;;
  render-config) render_config ;;
  print-env)     cmd_print_env ;;
  *)
    cat >&2 <<USAGE
用法: bash scripts/kvpool.sh <command>

  start         启动 mooncake_master（幂等；已有 master 则复用）
  stop          停止本脚本启动的 mooncake_master
  status        查看 master / 配置状态
  ensure        写配置并按需启动 master（serve.sh 调用）
  render-config 生成 mooncake.json 并打印路径
  print-env     打印生效配置
USAGE
    exit 2
    ;;
esac
