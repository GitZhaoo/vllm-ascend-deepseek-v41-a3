#!/usr/bin/env bash
# =============================================================================
# kill_vllm.sh — stop every vLLM instance on this node and clean NPU workers
#
# This node is expected to host one large model at a time. The script is
# deliberately thorough: it stops the project watchdog first (so nothing is
# restarted behind your back), then terminates vLLM main processes, their
# descendants and leftover NPU worker processes listed by npu-smi.
#
# Usage:
#   bash scripts/kill_vllm.sh                  # watchdog + all vLLM + workers
#   bash scripts/kill_vllm.sh --only-vllm      # keep watchdog running
#   bash scripts/kill_vllm.sh --dry-run        # show what would be done
#   PORT=8021 bash scripts/kill_vllm.sh        # only this port's process tree
#   PORT=8021 bash scripts/kill_vllm.sh --only-vllm --dry-run
#
# Environment:
#   MAX_ATTEMPTS=30  CHECK_INTERVAL=2  PORT=<port>
# =============================================================================
set -uo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
. "$SELF_DIR/common.sh"

SELF=$$
PORT_FILTER=${PORT:-}
ONLY_VLLM=0
DRY=0
VERBOSE=0
for a in "$@"; do
  case "$a" in
    --only-vllm)   ONLY_VLLM=1 ;;
    --dry-run)     DRY=1 ;;
    --verbose|-v)  VERBOSE=1 ;;
    --port)        PORT_FILTER="$2"; shift ;;
    -h|--help)     sed -n '2,25p' "$0"; exit 0 ;;
  esac
done

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
vlog() { [ "$VERBOSE" = 1 ] && log "$@" || true; }
proc_cmd() { [ -r "/proc/$1/cmdline" ] && tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null || true; }
do_kill() {
  local sig="$1"; shift
  if [ "$DRY" = 1 ]; then echo "    [dry-run] kill -$sig $*"; return 0; fi
  kill -"$sig" "$@" 2>/dev/null || true
}

# ---- 1) stop this project's watchdog(s) -----------------------------------
stop_watchdogs() {
  if [ "$ONLY_VLLM" = 1 ]; then vlog "跳过 watchdog（--only-vllm）"; return; fi
  local found=0 pid cmd p
  # Known pid file first.
  if [ -f "$DSV41_WATCHDOG_PID_FILE" ]; then
    pid=$(cat "$DSV41_WATCHDOG_PID_FILE" 2>/dev/null || true)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      log "停止 watchdog pid=$pid"
      do_kill TERM "$pid"; [ "$DRY" = 1 ] || sleep 1; do_kill KILL "$pid"
      found=1
    fi
    [ "$DRY" = 1 ] || rm -f "$DSV41_WATCHDOG_PID_FILE"
  fi
  # Fallback: any process running this project's watchdog.sh.
  for p in /proc/[0-9]*; do
    pid=${p#/proc/}
    [ "$pid" = "$SELF" ] && continue
    cmd=$(proc_cmd "$pid"); [ -n "$cmd" ] || continue
    case "$cmd" in
      *"watchdog.sh"*"run"*)
        log "停止 watchdog 进程 pid=$pid ($(echo "$cmd" | cut -c1-90))"
        do_kill TERM "$pid"; [ "$DRY" = 1 ] || sleep 1; do_kill KILL "$pid"
        found=1 ;;
    esac
  done
  [ "$found" = 0 ] && log "未发现运行中的本项目 watchdog"
}

# ---- stop prefix router ----------------------------------------------------
stop_router_process() {
  local pid
  pid=$(dsv41_find_router_pid || true)
  if [ -n "$pid" ]; then
    log "停止 prefix router pid=$pid"
    do_kill TERM "$pid"; [ "$DRY" = 1 ] || sleep 1; do_kill KILL "$pid"
  fi
  [ "$DRY" = 1 ] || rm -f "$DSV41_ROUTER_PID_FILE"
}

# ---- collect vLLM processes ------------------------------------------------
gather_descendants() {
  local root="$1" cur queue seen=" "
  queue="$root"
  while [ -n "${queue// /}" ]; do
    cur=${queue%% *}
    queue=${queue#"$cur"}; queue=${queue# }
    case "$seen" in *" $cur "*) continue ;; esac
    seen="$seen$cur "
    local c
    for c in $(ps -o pid= --ppid "$cur" 2>/dev/null); do queue="$queue $c"; done
  done
  echo $seen
}

collect_vllm_pids() {
  local pid comm cmd p
  if [ -n "$PORT_FILTER" ]; then
    for p in /proc/[0-9]*; do
      pid=${p#/proc/}
      [ "$pid" = "$SELF" ] && continue
      comm=$(cat "$p/comm" 2>/dev/null) || continue
      case "$comm" in python*|vllm) ;; *) continue ;; esac
      cmd=$(proc_cmd "$pid")
      case "$cmd" in
        *vllm.entrypoints*|*"vllm serve"*|*"-m vllm"*)
          if echo "$cmd" | grep -q -- "--port $PORT_FILTER"; then
            gather_descendants "$pid"
          elif echo "$cmd" | grep -qE -- "--data-parallel-supervisor-port ($PORT_FILTER|$((PORT_FILTER - 1)))"; then
            # external-LB: router 8021, supervisor 8020, rank APIs 8022+.
            gather_descendants "$pid"
          fi
          ;;
      esac
    done
    return
  fi
  # All vLLM engines on the machine.
  ps -eo pid=,comm= 2>/dev/null | awk '$2 ~ /^VLLM::/ {print $1}'
  for p in /proc/[0-9]*; do
    pid=${p#/proc/}
    [ "$pid" = "$SELF" ] && continue
    comm=$(cat "$p/comm" 2>/dev/null) || continue
    case "$comm" in python*|vllm) ;; *) continue ;; esac
    cmd=$(proc_cmd "$pid")
    case "$cmd" in
      *vllm.entrypoints*|*"vllm serve"*|*"-m vllm"*) echo "$pid" ;;
    esac
  done
}

# ---- 2) kill vLLM ----------------------------------------------------------
kill_vllm() {
  local pids alive pid
  pids=$(collect_vllm_pids | sort -u | tr '\n' ' ')
  if [ -z "${pids// /}" ]; then log "未发现运行中的 vLLM 进程"; return; fi
  log "停止 vLLM 进程: $pids"
  do_kill TERM $pids
  [ "$DRY" = 1 ] && return
  sleep 3
  alive=""
  for pid in $pids; do kill -0 "$pid" 2>/dev/null && alive="$alive $pid"; done
  if [ -n "${alive// /}" ]; then
    log "强制 KILL:$alive"
    kill -KILL $alive 2>/dev/null || true
  fi
  rm -f "$DSV41_SERVE_PID_FILE"
}

# ---- 3) clean NPU workers --------------------------------------------------
npu_worker_pids() {
  command -v npu-smi >/dev/null 2>&1 || return 0
  npu-smi info 2>/dev/null | grep "VLLMWorker_DP" \
    | awk -F'|' '{gsub(/[ \t]+/,"",$2); print $2}' | grep -E '^[0-9]+$' | sort -u
}

kill_npu_workers() {
  [ -n "$PORT_FILTER" ] && return 0   # port mode relies on the process tree
  local pids
  pids=$(npu_worker_pids | tr '\n' ' ')
  [ -n "${pids// /}" ] || return 0
  vlog "npu-smi 残留 worker: $pids"
  do_kill KILL $pids
}

# ---- 4) wait until clean ---------------------------------------------------
wait_clear() {
  local max=${MAX_ATTEMPTS:-30} i remain npu_left
  for i in $(seq 1 "$max"); do
    kill_npu_workers
    remain=$(collect_vllm_pids | sort -u | tr '\n' ' ')
    npu_left=""
    [ -z "$PORT_FILTER" ] && npu_left=$(npu_worker_pids | tr '\n' ' ')
    if [ -z "${remain// /}" ] && [ -z "${npu_left// /}" ]; then
      log "清理完成（无 vLLM 进程 / NPU worker 残留）"
      return 0
    fi
    vlog "第 $i/$max 次检查仍有残留: ${remain} npu=[${npu_left}]"
    sleep "${CHECK_INTERVAL:-2}"
  done
  log "⚠ ${max} 次后仍有残留，请手动检查: ps -eo pid,comm | grep VLLM ; npu-smi info"
  return 1
}

stop_watchdogs
stop_router_process
kill_vllm
if [ "$DRY" = 1 ]; then
  log "dry-run 结束（未做任何修改）"
  exit 0
fi
wait_clear
rc=$?

echo "=== 残留 VLLM:: 进程 ==="
ps -eo pid,comm 2>/dev/null | awk '$2 ~ /^VLLM::/ {print}' | head || true
echo "=== NPU VLLMWorker 数量 ==="
npu_worker_pids | wc -l
echo "=== 端口监听 ($DSV41_PORT) ==="
if command -v ss >/dev/null 2>&1; then
  ss -ltn 2>/dev/null | grep -E ":${PORT_FILTER:-$DSV41_PORT}([^0-9]|$)" || echo "(${PORT_FILTER:-$DSV41_PORT} 无监听)"
else
  netstat -tln 2>/dev/null | grep -E ":${PORT_FILTER:-$DSV41_PORT}([^0-9]|$)" || echo "(${PORT_FILTER:-$DSV41_PORT} 无监听)"
fi
exit $rc
