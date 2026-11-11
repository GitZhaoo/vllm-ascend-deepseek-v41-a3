#!/usr/bin/env bash
# run_config.sh <config-name> [overlay.env]
#
# Starts ONE experiment config of the DeepSeek-V4.1-Flash engine on this node,
# waits for readiness, runs the fixed bench suite, snapshots evidence, stops.
# Never touches config files inside the shared repo checkout: everything is
# passed through exported environment variables and a throw-away test copy of
# the repo at $EXP/repo-test.
set -uo pipefail

CFG=${1:?config name}
OVERLAY=${2:-}
EXP=${EXP:-/cache/exp}
REPO=$EXP/repo-test
ROOTFS=/cache/dsv41-rc2-rootfs
MODEL=/mnt/sfs/yzl/glm5_2/models/Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8
PY=$ROOTFS/usr/local/python3.12.13/bin/python3.12

mkdir -p "$EXP/logs" "$EXP/json" "$EXP/home/$CFG"
export HOME="$EXP/home/$CFG"
export DSV41_ROOTFS=$ROOTFS
export DSV41_MODEL=$MODEL
export DSV41_PORT=8021
export DSV41_HOST=0.0.0.0
export DSV41_DP=2
export DSV41_TP=8
export DSV41_LB_MODE=external
export DSV41_DP_SUPERVISOR_PORT=8020
export DSV41_DP_BASE_PORT=8022
export DSV41_ROUTER_POLICY=cost
export DSV41_ROUTER_MAX_SKEW=8
export DSV41_API_KEY=sk-exp-test
export DSV41_MAX_MODEL_LEN=1048576
export DSV41_MAX_NUM_BATCHED_TOKENS=16384
export DSV41_MAX_NUM_SEQS=32
export DSV41_GPU_MEM_UTIL=0.90
export DSV41_BLOCK_SIZE=128
export DSV41_ENABLE_PREFIX_CACHING=1
export DSV41_ENABLE_EXPERT_PARALLEL=1
export DSV41_ENABLE_PROMPT_TOKENS_DETAILS=1
export DSV41_OFFLINE=1
export DSV41_SPEC_TOKENS=5
export DSV41_USE_V2=1
export DSV41_KV_POOL=0
export DSV41_KV_EVENTS=1
export DSV41_KV_EVENTS_BASE_PORT=5557
export DSV41_RESULTS=$EXP/logs
export DSV41_LOG_DIR=$EXP/logs
export DSV41_PID_DIR=/tmp
export DSV41_ALLOW_MULTI=1
export DSV41_ENGINE_READY_TIMEOUT_S=36000
export DSV41_CHECK_INTERVAL=30

if [ -n "$OVERLAY" ] && [ -f "$OVERLAY" ]; then
  # shellcheck disable=SC1090
  set -a; . "$OVERLAY"; set +a
fi

echo "=== [$CFG] $(date '+%F %T') overlay=${OVERLAY:-none}"
echo "--- effective knobs: gmode=${DSV41_CUDAGRAPH_MODE:-?} static_kernel=${DSV41_STATIC_KERNEL:-?} npugraph_ex=${DSV41_NPUGRAPH_EX:-?} prefill_mc2=${DSV41_PREFILL_MC2:-?} fused_mc2=${DSV41_FUSED_MC2:-?} ms_shared=${DSV41_MULTISTREAM_SHARED_EXPERT:-?} seqs=${DSV41_MAX_NUM_SEQS} bat=${DSV41_MAX_NUM_BATCHED_TOKENS} kvpool=${DSV41_KV_POOL}"

# --- stop anything left from a previous config -------------------------------
STOP=1 bash "$REPO/scripts/serve.sh" >/dev/null 2>&1 || true
sleep 3
pkill -9 -f "vllm[.]entrypoints[.]cli[.]main serve" 2>/dev/null || true
pkill -9 -f "VLLM[:][:]" 2>/dev/null || true
pkill -9 -f "prefix_router[.]py" 2>/dev/null || true
pkill -9 -f "memcache[_]hybrid" 2>/dev/null || true
pkill -9 -f "multiprocessing.spawn" 2>/dev/null || true
# wait until the NPUs are actually released (crashed runs leave HCCL ports/NPU
# memory busy for a while -> next start dies with HCCL EI0019 bind errors)
for _ in $(seq 1 30); do
  used=$(npu-smi info 2>/dev/null | grep -oE "[0-9]+ / 65536" | awk '{print $1}' | sort -n | tail -1)
  [ -z "$used" ] && break
  if [ "$used" -lt 4000 ]; then break; fi
  sleep 5
done
sleep 5

# --- start -------------------------------------------------------------------
START_TS=$(date +%s)
bash "$REPO/scripts/serve.sh" > "$EXP/logs/$CFG.launcher.log" 2>&1
echo "--- launcher rc=$? after $(( $(date +%s) - START_TS ))s"

READY=0
# give the launcher time to spawn the engine process before judging liveness
for _ in $(seq 1 36); do
  pgrep -f "vllm[.]entrypoints[.]cli[.]main serve" >/dev/null 2>&1 && break
  sleep 5
done
MISS=0
for _ in $(seq 1 90); do
  code=$(curl -s --noproxy '*' -o /dev/null -w '%{http_code}' -m 3 \
    -H "Authorization: Bearer $DSV41_API_KEY" "http://127.0.0.1:$DSV41_PORT/health" 2>/dev/null)
  if [ "$code" = "200" ]; then READY=1; break; fi
  if [ "$code" = "503" ]; then
    echo "-- health 503 (ranks failed); waiting for launcher log root cause"
  fi
  if pgrep -f "vllm[.]entrypoints[.]cli[.]main serve" >/dev/null 2>&1; then
    MISS=0
  else
    MISS=$((MISS + 1))
  fi
  ELAPSED=$(( $(date +%s) - START_TS ))
  if [ "$MISS" -ge 3 ] && [ "$ELAPSED" -gt 180 ]; then
    echo "!!! engine process gone for ~30s after ${ELAPSED}s of startup"
    break
  fi
  sleep 10
done
STARTUP_S=$(( $(date +%s) - START_TS ))
echo "--- ready=$READY startup=${STARTUP_S}s"
if [ "$READY" != "1" ]; then
  echo "!!! [$CFG] engine did not become ready; tail of launcher log:"
  tail -40 "$EXP/logs/$CFG.launcher.log"
  exit 1
fi

# --- evidence ----------------------------------------------------------------
SERVE_LOG=$(ls -t "$EXP"/logs/serve_*.log 2>/dev/null | head -1)
{
  echo "config=$CFG overlay=$OVERLAY startup_s=$STARTUP_S"
  echo "serve_log=$SERVE_LOG"
  grep -E "Capturing CUDA graphs|ACL graph replay|capturing|NPU KV cache size|Free memory on device" "$SERVE_LOG" 2>/dev/null | tail -15
  grep -E "enable_prefill_mc2|enable_fused_mc2|static_kernel|npugraph_ex|ERROR|Traceback" "$SERVE_LOG" 2>/dev/null | tail -10
} > "$EXP/logs/$CFG.evidence.txt"

# warm-up so the prefix cache / graphs are hot before measuring
curl -s --noproxy '*' -m 300 -H "Authorization: Bearer $DSV41_API_KEY" -H 'Content-Type: application/json' \
  -d "{\"model\":\"DeepSeek-V4.1-Flash-1017\",\"prompt\":\"WARM $(date +%s) $(printf 'x%.0s' {1..64})\",\"max_tokens\":16,\"temperature\":0,\"stream\":true}" \
  "http://127.0.0.1:$DSV41_PORT/v1/completions" >/dev/null 2>&1 || true
sleep 3

# --- bench -------------------------------------------------------------------
export PYTHONHOME=$ROOTFS/usr/local/python3.12.13
export PYTHONPATH="$ROOTFS/usr/local/python3.12.13/lib/python3.12/site-packages"
export PATH=$ROOTFS/usr/local/python3.12.13/bin:$PATH
"$PY" "$REPO/extra-info/exp/bench_suite.py" \
  --base-url "http://127.0.0.1:$DSV41_PORT" \
  --model DeepSeek-V4.1-Flash-1017 \
  --api-key "$DSV41_API_KEY" \
  --max-tokens 256 --repeats 3 \
  --dual-concurrency 2,16,64 \
  --out "$EXP/json/$CFG.json" 2>&1 | tee "$EXP/logs/$CFG.bench.log"

# resource snapshot while idle-ish right after bench
npu-smi info > "$EXP/logs/$CFG.npu.txt" 2>&1 || true

curl -s --noproxy '*' "http://127.0.0.1:$DSV41_PORT/metrics" > "$EXP/logs/$CFG.metrics.txt" 2>&1 || true

# --- stop --------------------------------------------------------------------
STOP=1 bash "$REPO/scripts/serve.sh" >/dev/null 2>&1 || true
sleep 3
python3 - "$EXP/json/$CFG.json" "$CFG" "$STARTUP_S" <<'PYEOF' >> "$EXP/summary.csv" 2>/dev/null || true
import json, sys
path, cfg, startup = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(path))
h = d.get("single_hot", {})
c = d.get("single_cold", {})
dual = {str(row["concurrency"]): row for row in d.get("dual", [])}
cols = [cfg, startup,
        f"{h.get('decode_tps',0):.1f}", f"{h.get('ttft_ms',0):.0f}",
        f"{h.get('ms_per_step',0):.2f}", f"{h.get('acceptance') or 0:.3f}",
        f"{c.get('ttft_ms',0):.0f}",
        f"{dual.get('2',{}).get('aggregate',0):.1f}",
        f"{dual.get('16',{}).get('aggregate',0):.1f}",
        f"{dual.get('64',{}).get('aggregate',0):.1f}"]
print(",".join(str(x) for x in cols))
PYEOF
echo "=== [$CFG] done $(date '+%F %T')"
