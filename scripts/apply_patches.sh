#!/usr/bin/env bash
# =============================================================================
# apply_patches.sh — apply the DeepSeek-V4.1 patch series
#
# Two source trees are managed independently:
#   * vllm-ascend (performance / KV-pool correctness)
#   * vllm        (DeepSeek-V4.1 tokenizer input safety)
#
# Every marker stores the SHA-256 of the patch that produced it. A changed
# patch is therefore reported as "changed" instead of being silently treated as
# already applied (the previous marker-only check could hide a stale tree).
#
# Usage:
#   bash scripts/apply_patches.sh status
#   bash scripts/apply_patches.sh check
#   bash scripts/apply_patches.sh apply       # idempotent
#   bash scripts/apply_patches.sh verify
#   bash scripts/apply_patches.sh revert --force
#
# Targets can be overridden (single-component mode):
#   SRC=/path/to/vllm-ascend PATCH_DIR=/path/to/patches \
#     bash scripts/apply_patches.sh apply
# =============================================================================
set -Eeuo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
. "$SELF_DIR/common.sh"

# The patched sources live inside the image: with DSV41_RUNTIME=docker run
# everything in the container (no-op on the native runtime / inside the container).
dsv41_reexec_in_container "$SELF_DIR/apply_patches.sh" "$@"

GATES=(
  VLLM_ASCEND_ROPE_INDEX_SELECT
  VLLM_ASCEND_V41_TRITON_SLOT_MAPPING
  VLLM_ASCEND_V41_MOE_MASK_RANGE
  VLLM_ASCEND_V41_O_PROJ_2D
  VLLM_ASCEND_V41_QLI_NO_CANDIDATE
  VLLM_ASCEND_V41_ENGRAM_WKV_TP
)

# ---- components ------------------------------------------------------------
COMP_NAMES=()
COMP_SRCS=()
COMP_PATCH_DIRS=()

if [ -n "${SRC:-}" ] || [ -n "${PATCH_DIR:-}" ]; then
  COMP_NAMES=(custom)
  COMP_SRCS=("${SRC:-${DSV41_ROOTFS%/}/vllm-workspace/vllm-ascend}")
  COMP_PATCH_DIRS=("${PATCH_DIR:-$SELF_DIR/patches/vllm-ascend-v41-rc2}")
else
  COMP_NAMES=(vllm-ascend vllm)
  COMP_SRCS=(
    "${DSV41_ROOTFS%/}/vllm-workspace/vllm-ascend"
    "${DSV41_ROOTFS%/}/vllm-workspace/vllm"
  )
  COMP_PATCH_DIRS=(
    "$SELF_DIR/patches/vllm-ascend-v41-rc2"
    "$SELF_DIR/patches/vllm-ced6857af"
  )
fi

# Current component (set by select_component).
C_NAME=""
SRC=""
PATCH_DIR=""
SERIES=""
MARKER_DIR=""

select_component() {
  local i="$1"
  C_NAME=${COMP_NAMES[$i]}
  SRC=${COMP_SRCS[$i]}
  PATCH_DIR=${COMP_PATCH_DIRS[$i]}
  SERIES="$PATCH_DIR/series.conf"
  MARKER_DIR="$SRC/.patches-applied.d"
}

component_indices() { seq 0 $((${#COMP_NAMES[@]} - 1)); }

patches() { grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$SERIES"; }
is_git_tree() { git -C "$SRC" rev-parse --is-inside-work-tree >/dev/null 2>&1; }
marker_for() { echo "$MARKER_DIR/$1.done"; }
fingerprint_file() { echo "$MARKER_DIR/series.fingerprint"; }

patch_sha() { sha256sum "$PATCH_DIR/$1" | awk '{print $1}'; }

marker_sha() {
  local m
  m=$(marker_for "$1")
  [ -f "$m" ] || return 1
  sed -n 's/^sha256=//p' "$m" | head -n 1
}

write_marker() {
  local f="$1" tmp
  mkdir -p "$MARKER_DIR"
  tmp=$(mktemp "$MARKER_DIR/.marker.XXXXXX")
  {
    printf 'sha256=%s\n' "$(patch_sha "$f")"
    printf 'applied=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  } > "$tmp"
  mv -f "$tmp" "$(marker_for "$f")"
}

content_state() {
  python3 "$SELF_DIR/patch_utils.py" content-state \
    --src "$SRC" --patch "$PATCH_DIR/$1" 2>/dev/null || echo unknown
}

series_fingerprint() {
  python3 - "$SERIES" "$PATCH_DIR" "$SRC" <<'PY'
import hashlib
import os
import subprocess
import sys

series, patch_dir, src = sys.argv[1:4]
h = hashlib.sha256()
with open(series, "rb") as fh:
    h.update(fh.read())
names = []
with open(series, encoding="utf-8") as fh:
    for raw in fh:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line)
for name in names:
    h.update(name.encode())
    path = os.path.join(patch_dir, name)
    if not os.path.exists(path):
        h.update(b"MISSING")
        continue
    with open(path, "rb") as fh:
        h.update(fh.read())
try:
    head = subprocess.check_output(
        ["git", "-C", src, "rev-parse", "HEAD"],
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()
except (OSError, subprocess.CalledProcessError):
    head = "unknown"
h.update(head.encode())
print(h.hexdigest())
PY
}

write_series_fingerprint() {
  local tmp
  mkdir -p "$MARKER_DIR"
  tmp=$(mktemp "$MARKER_DIR/.fingerprint.XXXXXX")
  series_fingerprint > "$tmp"
  mv -f "$tmp" "$(fingerprint_file)"
}

# Print one of: applied | applicable | changed | conflict | unknown
patch_state() {
  local f="$1" content m want got
  m=$(marker_for "$f")
  if [ -f "$m" ]; then
    want=$(patch_sha "$f")
    got=$(marker_sha "$f" || true)
    if [ -n "$got" ] && [ "$got" != "$want" ]; then
      echo changed
      return
    fi
    if [ -z "$got" ]; then
      # Legacy marker (pre-SHA). Trust the tree only if it still contains the
      # patch, otherwise the patch may have changed since the marker was made.
      if is_git_tree && git -C "$SRC" apply --reverse --check "$PATCH_DIR/$f" >/dev/null 2>&1; then
        echo applied
        return
      fi
      content=$(content_state "$f")
      if [ "$content" = "applied" ]; then
        echo applied
      else
        echo changed
      fi
      return
    fi
    echo applied
    return
  fi

  if is_git_tree; then
    if git -C "$SRC" apply --check "$PATCH_DIR/$f" >/dev/null 2>&1; then
      echo applicable
      return
    fi
    if git -C "$SRC" apply --reverse --check "$PATCH_DIR/$f" >/dev/null 2>&1; then
      echo applied
      return
    fi
    # Gate-default pinning changes lines introduced by the patch, so the
    # reverse check can fail even when the patch is fully applied.
    content=$(content_state "$f")
    case "$content" in
      applied) echo applied ;;
      partial) echo conflict ;;
      *)       echo conflict ;;
    esac
    return
  fi

  content=$(content_state "$f")
  if [ "$content" = "applied" ]; then
    echo applied
  elif [ "$content" = "partial" ]; then
    echo conflict
  elif command -v patch >/dev/null 2>&1 \
       && patch -p1 -R --dry-run --batch -d "$SRC" < "$PATCH_DIR/$f" >/dev/null 2>&1; then
    echo applied
  else
    echo unknown
  fi
}

apply_one() {
  local f="$1"
  if is_git_tree; then
    git -C "$SRC" apply "$PATCH_DIR/$f"
  elif command -v patch >/dev/null 2>&1; then
    patch -p1 -d "$SRC" --forward --batch < "$PATCH_DIR/$f"
  else
    echo "[apply] ❌ 无法应用 $f（既不是 git 树，也没有 patch 命令）" >&2
    return 1
  fi
  write_marker "$f"
}

ensure_marker() {
  local f="$1" got
  got=$(marker_sha "$f" || true)
  if [ -z "$got" ] || [ "$got" != "$(patch_sha "$f")" ]; then
    write_marker "$f"
  fi
}

# Pin gate defaults to 1 (idempotent).
fix_gate_defaults() {
  python3 - "$SRC" <<'PY'
import re
import os
import sys

src = sys.argv[1]
env_gates = [
    "VLLM_ASCEND_ROPE_INDEX_SELECT",
    "VLLM_ASCEND_V41_TRITON_SLOT_MAPPING",
    "VLLM_ASCEND_V41_MOE_MASK_RANGE",
    "VLLM_ASCEND_V41_O_PROJ_2D",
    "VLLM_ASCEND_V41_QLI_NO_CANDIDATE",
    "VLLM_ASCEND_V41_ENGRAM_WKV_TP",
]
mod_gates = {
    "vllm_ascend/models/deepseek_v41/engram/layer.py": "VLLM_ASCEND_V41_ENGRAM_WKV_TP",
    # model.py has its own independent read of the same gate (weight sharding).
    "vllm_ascend/models/deepseek_v41/model.py": "VLLM_ASCEND_V41_ENGRAM_WKV_TP",
    "vllm_ascend/models/deepseek_v41/indexer.py": "VLLM_ASCEND_V41_QLI_NO_CANDIDATE",
    "vllm_ascend/ops/fused_moe/token_dispatcher.py": "VLLM_ASCEND_V41_MOE_MASK_RANGE",
}
changed = []

p = f"{src}/vllm_ascend/envs.py"
s = open(p, encoding="utf-8").read()
origin = s
for gate in env_gates:
    pat = re.compile(r'_strict_binary_env\(\s*"%s"\s*\)' % re.escape(gate))
    s2 = pat.sub(f'_strict_binary_env("{gate}", "1")', s)
    if s2 != s:
        changed.append(f"envs.py:{gate}")
        s = s2
if s != origin:
    open(p, "w", encoding="utf-8").write(s)

for rel, gate in mod_gates.items():
    p = f"{src}/{rel}"
    if not os.path.exists(p):
        print(f"[apply] ⚠ 跳过不存在的文件: {rel}")
        continue
    s = open(p, encoding="utf-8").read()
    pat = re.compile(r'os\.environ\.get\(\s*"%s"(?:\s*,\s*"[01]")?\s*\)' % re.escape(gate))
    s2 = pat.sub(f'os.environ.get("{gate}", "1")', s)
    if s2 != s:
        open(p, "w", encoding="utf-8").write(s2)
        changed.append(rel)

if changed:
    print("[apply] gate 默认值已改为 1: " + ", ".join(changed))
else:
    print("[apply] gate 默认值已是 1，无需改动")
PY
}

component_ready() {
  case "$C_NAME" in
    vllm-ascend) [ -f "$SRC/vllm_ascend/envs.py" ] ;;
    vllm)        [ -f "$SRC/vllm/__init__.py" ] ;;
    *)           [ -d "$SRC" ] ;;
  esac
}

apply_component() {
  local f state
  if [ ! -f "$SERIES" ]; then
    echo "[apply] ❌ 缺少补丁序列: $SERIES" >&2
    exit 1
  fi
  if ! component_ready; then
    echo "[apply] ❌ 目标不存在或不是预期的源码树: $SRC" >&2
    echo "[apply]    先运行 bash scripts/setup_rootfs.sh" >&2
    exit 1
  fi
  for f in $(patches); do
    state=$(patch_state "$f")
    case "$state" in
      applied)
        echo "[apply] 已应用，跳过 $f"
        ensure_marker "$f"
        ;;
      applicable|unknown)
        echo "[apply] applying $f"
        apply_one "$f"
        ;;
      changed)
        echo "[apply] ❌ $f 的补丁内容已变化，但部署树里还是旧版本。" >&2
        echo "[apply]    先回滚再重新应用（不要手工删 marker）:" >&2
        echo "[apply]      bash scripts/apply_patches.sh revert --force" >&2
        echo "[apply]      bash scripts/apply_patches.sh apply" >&2
        exit 1
        ;;
      conflict)
        echo "[apply] ❌ $f 与当前源码冲突（既不能正向应用，也不能反向应用）。" >&2
        echo "[apply]    可能已手工修改相关文件，或 rootfs 版本不匹配。" >&2
        echo "[apply]    排查: git -C \"$SRC\" apply --check \"$PATCH_DIR/$f\"" >&2
        exit 1
        ;;
    esac
  done
  if [ "$C_NAME" = "vllm-ascend" ]; then
    fix_gate_defaults
  fi
  write_series_fingerprint
}

apply_patches() {
  local i
  for i in $(component_indices); do
    select_component "$i"
    echo "[apply] === $C_NAME: $SRC"
    apply_component
  done
  echo "[apply] 完成。重启服务生效: bash scripts/watchdog.sh restart"
}

check_component() {
  local rc=0 f state fp
  fp=$(fingerprint_file)
  if [ -f "$fp" ]; then
    if [ "$(cat "$fp")" != "$(series_fingerprint)" ]; then
      echo "  [序列已变化]   补丁序列/补丁内容/基线镜像与上次 apply 不同"
      echo "                 需 revert --force 后重新 apply"
      rc=1
    fi
  else
    echo "  [指纹未记录]   下次 apply 会写入（旧版 marker 可兼容读取）"
  fi
  for f in $(patches); do
    state=$(patch_state "$f")
    case "$state" in
      applicable) echo "  [可应用]       $f" ;;
      applied)    echo "  [已应用]       $f" ;;
      changed)    echo "  [补丁已变更]   $f（需 revert+apply）"; rc=1 ;;
      unknown)    echo "  [无法判定]     $f（非 git 树且反向 dry-run 失败）"; rc=1 ;;
      conflict)   echo "  [冲突/需处理]  $f"; rc=1 ;;
    esac
  done
  return $rc
}

check_patches() {
  local rc=0 i
  for i in $(component_indices); do
    select_component "$i"
    echo "--- $C_NAME: $SRC"
    check_component || rc=1
  done
  return $rc
}

revert_all() {
  local force="${2:-}"
  local i f files=() p
  for i in $(component_indices); do
    select_component "$i"
    echo "[revert] $C_NAME: $SRC"
    if ! is_git_tree; then
      echo "[revert] ❌ $SRC 不是 git 工作树，无法自动回滚；请重新解包 rootfs" >&2
      return 1
    fi
    files=()
    while IFS= read -r p; do
      [ -n "$p" ] && files+=("$p")
    done < <(
      for f in $(patches); do
        python3 "$SELF_DIR/patch_utils.py" files --patch "$PATCH_DIR/$f"
      done | sort -u
    )
    if [ "${#files[@]}" -eq 0 ]; then
      echo "[revert] 没有可回滚的文件"
      continue
    fi
    echo "[revert] 将恢复以下文件到 git HEAD（丢弃补丁与 gate 改动）:"
    printf '    %s\n' "${files[@]}"
    echo
    if [ "$force" != "--force" ]; then
      echo "❌ 该操作会丢弃这些文件的本地改动。确认请加 --force"
      return 1
    fi
    git -C "$SRC" checkout -- "${files[@]}"
    rm -rf "$MARKER_DIR"
    echo "[revert] 完成"
  done
}

status() {
  local i rc=0
  for i in $(component_indices); do
    select_component "$i"
    echo "COMPONENT : $C_NAME"
    echo "SRC       : $SRC"
    echo "PATCH_DIR : $PATCH_DIR"
    if [ "$C_NAME" = "vllm-ascend" ]; then
      if [ -f "$SRC/vllm_ascend/envs.py" ] && grep -q "VLLM_ASCEND_V41_TRITON_SLOT_MAPPING" "$SRC/vllm_ascend/envs.py"; then
        echo "patches   : ✅ 已应用"
      else
        echo "patches   : ❌ 未应用"
      fi
    elif [ "$C_NAME" = "vllm" ]; then
      if grep -q "def escape_control_tokens" "$SRC/vllm/tokenizers/deepseek_v41_encoding.py" 2>/dev/null; then
        echo "patches   : ✅ 已应用"
      else
        echo "patches   : ❌ 未应用"
      fi
    fi
    echo "--- 每个补丁 ---"
    check_component || rc=1
    if [ "$C_NAME" = "vllm-ascend" ]; then
      echo "--- gate 默认值 ---"
      python3 - "$SRC" "${GATES[@]}" <<'PY'
import re
import sys

src = sys.argv[1]
gates = sys.argv[2:]
path = f"{src}/vllm_ascend/envs.py"
try:
    s = open(path, encoding="utf-8").read()
except OSError:
    print("  (envs.py 不存在)")
    raise SystemExit(0)
for gate in gates:
    m = re.search(r'_strict_binary_env\(\s*"%s"(?:\s*,\s*"(\d)")?\s*\)' % re.escape(gate), s)
    if not m:
        print(f"  {gate:<42} (未定义)")
    else:
        print(f"  {gate:<42} default={m.group(1) or '0'}")
PY
    fi
    echo
  done
  verify_patches || rc=1
  return $rc
}

verify_gates() {
  local rc=0
  python3 - "$SRC" "${GATES[@]}" <<'PY' || rc=$?
import os
import re
import sys

src, gates = sys.argv[1], sys.argv[2:]
env_path = f"{src}/vllm_ascend/envs.py"
s = open(env_path, encoding="utf-8").read()
problems = []
for gate in gates:
    m = re.search(r'_strict_binary_env\(\s*"%s"(?:\s*,\s*"(\d)")?\s*\)' % re.escape(gate), s)
    if not m:
        problems.append(f"{gate}: envs.py 未定义")
    elif (m.group(1) or "0") != "1":
        problems.append(f"{gate}: envs.py default={m.group(1) or '0'}")

module_gates = {
    "vllm_ascend/models/deepseek_v41/engram/layer.py": "VLLM_ASCEND_V41_ENGRAM_WKV_TP",
    "vllm_ascend/models/deepseek_v41/model.py": "VLLM_ASCEND_V41_ENGRAM_WKV_TP",
    "vllm_ascend/models/deepseek_v41/indexer.py": "VLLM_ASCEND_V41_QLI_NO_CANDIDATE",
    "vllm_ascend/ops/fused_moe/token_dispatcher.py": "VLLM_ASCEND_V41_MOE_MASK_RANGE",
}
for rel, gate in module_gates.items():
    path = f"{src}/{rel}"
    if not os.path.exists(path):
        problems.append(f"{rel}: 文件不存在")
        continue
    text = open(path, encoding="utf-8").read()
    m = re.search(r'os\.environ\.get\(\s*"%s"(?:\s*,\s*"([01])")?\s*\)' % re.escape(gate), text)
    if not m:
        problems.append(f"{rel}: 未读取 {gate}")
    elif (m.group(1) or "0") != "1":
        problems.append(f"{rel}: {gate} default={m.group(1) or '0'}")

if problems:
    print("[verify] ❌ " + "; ".join(problems))
    raise SystemExit(1)
print("[verify] ✅ 6 个 gate 已定义且默认值为 1（含模块级读取）")
PY
  return $rc
}

verify_patches() {
  local i rc=0 ok=1 checked=0
  for i in $(component_indices); do
    select_component "$i"
    case "$C_NAME" in
      vllm-ascend)
        checked=$((checked + 1))
        verify_gates || rc=1
        if ! grep -q '_preempted_req_ids.discard(request.req_id)' \
             "$SRC/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py" 2>/dev/null; then
          echo "[verify] ❌ 0101 KV 池 MRV2 抢占修复未应用"
          rc=1; ok=0
        fi
        ;;
      vllm)
        checked=$((checked + 1))
        enc="$SRC/vllm/tokenizers/deepseek_v41_encoding.py"
        if ! grep -q 'def escape_control_tokens' "$enc" 2>/dev/null; then
          echo "[verify] ❌ vllm tokenizer 控制 token 逃逸修复未应用"
          rc=1; ok=0
        fi
        if grep -q 'assert content, f"Invalid message for role' "$enc" 2>/dev/null; then
          echo "[verify] ❌ vllm tokenizer 空 developer 消息修复未应用（仍 assert content）"
          rc=1; ok=0
        fi
        ;;
    esac
  done
  if [ "$checked" -eq 0 ]; then
    echo "[verify] （custom 单组件模式，跳过内置内容校验）"
  elif [ "$ok" = 1 ]; then
    echo "[verify] ✅ 补丁内容校验通过（vllm-ascend gates + 0101 + vllm encoding）"
  fi
  return $rc
}

case "${1:-status}" in
  apply)  apply_patches ;;
  check)  check_patches ;;
  verify) verify_patches ;;
  revert) revert_all "$@" ;;
  status) status ;;
  *) echo "用法: $0 {status|check|apply|verify|revert --force}"; exit 1 ;;
esac
