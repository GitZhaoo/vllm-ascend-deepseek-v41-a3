#!/usr/bin/env bash
# Build a PGO-optimised CPython 3.12.13 for the DSV41 rootfs.
#
# Motivation: reference deployment (repo1) measured -4.4% decode ms/step with a
# PGO libpython in the same会话 A/B. The rootfs python is built with
# `--enable-shared` and no LTO, so we reproduce that configuration.
#
# Usage:
#   bash scripts/experiments/build_pgo_python.sh /cache/pgo-src.tgz /cache/pgo
# Then swap /cache/pgo/py312/lib/libpython3.12.so.1.0 over
#   <rootfs>/usr/local/python3.12.13/lib/libpython3.12.so.1.0
# (keep a .orig copy; `scripts/experiments/run_config.sh` does this automatically
#  when the overlay sets EXP_PGO_SWAP=1).
set -euo pipefail
SRC_TGZ=${1:?path to Python-3.12.13.tgz}
WORK=${2:-/cache/pgo}
mkdir -p "$WORK"
cd "$WORK"
tar xf "$SRC_TGZ"
cd Python-3.12.13
./configure --prefix="$WORK/py312" --enable-shared --enable-optimizations \
  --disable-test-modules --with-ensurepip=no
make -j"$(nproc)"
make install
PYTHONHOME="$WORK/py312" "$WORK/py312/bin/python3.12" -c "import sys;print('PGO_PY_OK', sys.version)"
echo "swap with: cp $WORK/py312/lib/libpython3.12.so.1.0 <rootfs>/usr/local/python3.12.13/lib/"
