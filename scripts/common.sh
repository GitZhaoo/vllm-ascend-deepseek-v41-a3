#!/usr/bin/env bash
# shellcheck shell=bash
# =============================================================================
# common.sh — shared defaults and helpers for vllm-ascend-deepseek-v41-a3
#
# Source this file from other scripts:
#     . "$(dirname "$0")/common.sh"
#
# Precedence (high → low):
#   1. exported environment variables
#   2. config/dsv41.local.env
#   3. config/dsv41.env
#   4. defaults below
# =============================================================================

if [ -n "${DSV41_COMMON_SH_LOADED:-}" ]; then
  return 0 2>/dev/null || exit 0
fi
DSV41_COMMON_SH_LOADED=1

DSV41_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DSV41_PROJECT_ROOT

# Load configuration files. They are plain shell files with KEY=VALUE lines.
# Exported DSV41_* variables keep precedence over file values: snapshot them
# first, source the files, then restore the snapshot.
_dsv41_env_backup=$(mktemp "${TMPDIR:-/tmp}/dsv41-env.XXXXXX")
export -p | grep -E '^declare -x DSV41_[A-Za-z0-9_]+=' > "$_dsv41_env_backup" || true
for _dsv41_cfg in \
  "$DSV41_PROJECT_ROOT/config/dsv41.env" \
  "$DSV41_PROJECT_ROOT/config/dsv41.local.env"; do
  if [ -f "$_dsv41_cfg" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$_dsv41_cfg"
    set +a
  fi
done
if [ -s "$_dsv41_env_backup" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$_dsv41_env_backup"
  set +a
fi
rm -f "$_dsv41_env_backup"
unset _dsv41_cfg _dsv41_env_backup

# ---- paths -----------------------------------------------------------------
: "${DSV41_ROOTFS:=/cache/dsv41-rc2-rootfs}"
: "${DSV41_MODEL:=$DSV41_PROJECT_ROOT/models/DeepSeek-V4.1-Flash-1017-w8a8}"
: "${DSV41_RESULTS:=$DSV41_PROJECT_ROOT/logs}"
: "${DSV41_LOG_DIR:=$DSV41_PROJECT_ROOT/logs}"
: "${DSV41_PID_DIR:=/tmp}"

# ---- service ---------------------------------------------------------------
: "${DSV41_PORT:=8021}"
: "${DSV41_HOST:=0.0.0.0}"
: "${DSV41_SERVED_NAME:=DeepSeek-V4.1-Flash-1017}"
: "${DSV41_DP:=2}"
: "${DSV41_TP:=8}"

# API key. Accept the legacy name as a fallback but prefer DSV41_API_KEY.
: "${DSV41_API_KEY-${DSV41_W8A8_API_KEY:-}}"

# ---- pid files -------------------------------------------------------------
: "${DSV41_SERVE_PID_FILE:=$DSV41_PID_DIR/dsv41_serve.pid}"
: "${DSV41_WATCHDOG_PID_FILE:=$DSV41_PID_DIR/dsv41_watchdog.pid}"
: "${DSV41_ROUTER_PID_FILE:=$DSV41_PID_DIR/dsv41_router.pid}"
: "${DSV41_DP_BASE_PORT:=$((DSV41_PORT + 1))}"
: "${DSV41_DP_SUPERVISOR_PORT:=$((DSV41_PORT - 1))}"

export DSV41_ROOTFS DSV41_MODEL DSV41_RESULTS DSV41_LOG_DIR DSV41_PID_DIR
export DSV41_PORT DSV41_HOST DSV41_SERVED_NAME DSV41_DP DSV41_TP
export DSV41_API_KEY DSV41_SERVE_PID_FILE DSV41_WATCHDOG_PID_FILE DSV41_ROUTER_PID_FILE
export DSV41_DP_BASE_PORT DSV41_DP_SUPERVISOR_PORT

# ---- helpers ---------------------------------------------------------------

# Health checks and local API calls must never go through a corporate proxy.
dsv41_unset_proxy() {
  unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
  export no_proxy="127.0.0.1,localhost,::1,${no_proxy:-}"
  export NO_PROXY="$no_proxy"
}

dsv41_log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

# True when a cmdline belongs to this deployment's vLLM server. External-LB
# mode runs a multi-port supervisor whose command line has no --port 8021.
dsv41_is_serve_cmd() {
  case "$1" in
    *vllm.entrypoints.cli.main*"--port $DSV41_PORT"*) return 0 ;;
    *vllm.entrypoints.cli.main*"--data-parallel-multi-port-external-lb"*"--data-parallel-supervisor-port $DSV41_DP_SUPERVISOR_PORT"*) return 0 ;;
  esac
  return 1
}

# Return the PID of the vLLM server / DP supervisor for this deployment.
# Prefer the pid file; fall back to scanning /proc for the entrypoint command.
dsv41_find_serve_pid() {
  local pid="" p cmd
  if [ -f "$DSV41_SERVE_PID_FILE" ]; then
    pid=$(cat "$DSV41_SERVE_PID_FILE" 2>/dev/null || true)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
      if dsv41_is_serve_cmd "$cmd"; then echo "$pid"; return 0; fi
    fi
  fi
  for p in /proc/[0-9]*; do
    pid=${p#/proc/}
    [ -r "$p/cmdline" ] || continue
    cmd=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null) || continue
    if dsv41_is_serve_cmd "$cmd"; then echo "$pid"; return 0; fi
  done
  return 1
}

dsv41_find_router_pid() {
  local pid="" p cmd
  if [ -f "$DSV41_ROUTER_PID_FILE" ]; then
    pid=$(cat "$DSV41_ROUTER_PID_FILE" 2>/dev/null || true)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
      case "$cmd" in *prefix_router.py*"--listen $DSV41_PORT"*) echo "$pid"; return 0 ;; esac
    fi
  fi
  for p in /proc/[0-9]*; do
    pid=${p#/proc/}
    [ -r "$p/cmdline" ] || continue
    cmd=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null) || continue
    case "$cmd" in *prefix_router.py*"--listen $DSV41_PORT"*) echo "$pid"; return 0 ;; esac
  done
  return 1
}

# Kill the process tree rooted at $1 (children first is not required for KILL).
dsv41_collect_descendants() {
  local root="$1" cur queue seen=" "
  queue="$root"
  while [ -n "${queue// /}" ]; do
    cur=${queue%% *}
    queue=${queue#"$cur"}; queue=${queue# }
    case "$seen" in *" $cur "*) continue ;; esac
    seen="$seen$cur "
    local c
    for c in $(ps -o pid= --ppid "$cur" 2>/dev/null); do
      queue="$queue $c"
    done
  done
  echo "$seen"
}

# Best-effort cleanup of this instance's main process, descendants and
# (optionally) any leftover VLLM:: worker processes on the machine.
# Set DSV41_CLEAN_ALL_VLLM=0 when sharing the node with other vLLM services.
dsv41_cleanup_residual() {
  local pid pids=""
  pid=$(dsv41_find_serve_pid || true)
  if [ -n "$pid" ]; then
    pids="$(dsv41_collect_descendants "$pid") $pid"
  fi
  if [ "${DSV41_CLEAN_ALL_VLLM:-1}" = "1" ]; then
    local extra
    extra=$(ps -eo pid=,comm= 2>/dev/null | awk '$2 ~ /^VLLM::/ {print $1}')
    pids="$pids $extra"
  fi
  pids=$(echo $pids | tr ' ' '\n' | sort -u | tr '\n' ' ')
  if [ -n "${pids// /}" ]; then
    dsv41_log "清理残留 vLLM 进程: $pids"
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
  fi
  sleep 3
  rm -f "$DSV41_SERVE_PID_FILE"
}
