#!/usr/bin/env bash
# =============================================================================
# import_rootfs.sh — 从本地 tar 包导入 rootfs（无 registry 访问时的离线路径）
#
# 适用场景：离线机器 / 无法访问 quay.nju.edu.cn / 直接复制已有节点的 rootfs。
#
# 用法:
#   bash scripts/import_rootfs.sh <rootfs.tar> [--dest DIR] [--manifest REF.tsv]
#
#   --dest DIR        解包目标（默认 $DSV41_ROOTFS，必须为空或不存在）
#   --manifest REF    （可选）参考 manifest，解包后用 scripts/verify_rootfs.py 比对
#                     期望：OK: manifests identical
#
# 生成参考 manifest（在健康节点上执行）:
#   python3 scripts/verify_rootfs.py manifest /cache/dsv41-rc2-rootfs -o ref.tsv
#
# 注意：tar 经共享盘/网络中转时曾出现符号链接被展开成 0 字节文件、库被截断的问题；
# 这类损坏会让引擎以 HCCL/算子加载错误的形式失败（EI0015 / ERR00100），
# 所以离线导入后**务必**跑一次 --manifest 校验。
# =============================================================================
set -Eeuo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SELF_DIR/.." && pwd)
# shellcheck source=common.sh
. "$SELF_DIR/common.sh"

TARBALL=${1:?usage: import_rootfs.sh <rootfs.tar> [--dest DIR] [--manifest REF.tsv]}
shift || true
DEST=${DSV41_ROOTFS%/}
MANIFEST=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dest) DEST=${2%/}; shift 2 ;;
    --manifest) MANIFEST=$2; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[ -f "$TARBALL" ] || { echo "❌ tar 不存在: $TARBALL" >&2; exit 1; }
if [ -d "$DEST" ] && [ -n "$(ls -A "$DEST" 2>/dev/null)" ]; then
  echo "❌ 目标目录非空: $DEST（先手动移走或改用 --dest）" >&2; exit 1
fi

echo "[import] tar=$TARBALL -> dest=$DEST"
mkdir -p "$DEST"
tar -xf "$TARBALL" -C "$DEST" --strip-components=1
echo "[import] 解包完成: $(du -sh "$DEST" | cut -f1)"

PY="$DEST/usr/local/python3.12.13/bin/python3.12"
if [ -x "$PY" ]; then
  PYTHONHOME="$DEST/usr/local/python3.12.13" "$PY" -V && echo "[import] rootfs python ✅"
else
  echo "⚠️ 未找到 rootfs python（$PY），请检查 tar 结构与 --strip-components" >&2
fi

if [ -n "$MANIFEST" ]; then
  echo "[import] 校验 manifest: $MANIFEST"
  python3 "$SELF_DIR/verify_rootfs.py" manifest "$DEST" -o /tmp/rootfs_imported.tsv
  if python3 "$SELF_DIR/verify_rootfs.py" verify "$MANIFEST" /tmp/rootfs_imported.tsv; then
    echo "[import] ✅ rootfs 与参考 manifest 一致"
  else
    echo "[import] ❌ rootfs 与参考 manifest 不一致（见上面的差异）；不要用于部署" >&2
    exit 3
  fi
else
  echo "[import] ⚠️ 未提供 --manifest，跳过完整性校验（强烈建议补上）"
fi

echo "[import] 后续: 确认 config/dsv41.local.env 的 DSV41_ROOTFS=$DEST"
echo "        然后: bash scripts/apply_patches.sh apply && bash scripts/doctor.sh"
