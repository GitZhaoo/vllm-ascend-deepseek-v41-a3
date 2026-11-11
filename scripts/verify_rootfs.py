#!/usr/bin/env python3
"""Generate / verify a manifest of an extracted Ascend rootfs.

Why this exists
---------------
The deployment ships the rc2 image as a *rootfs directory* (`setup_rootfs.sh`
unpacks OCI layers). Copying that directory between nodes with `tar` over a
shared filesystem silently produced a broken tree in our 2026-11-11 test run:

  * symlinks became 0-byte regular files
    (`cann-9.2.0/aarch64-linux/lib64/libccl_dpu.so` -> regular 0-byte file),
  * one library was truncated (`libhccl_alg.so` 9.5 MB instead of 12.1 MB).

The engine then died during worker init with misleading errors
(`ImportError: libccl_dpu.so: file too short`, `ERR00100 aclrtBinaryLoadFromData`,
`EI0015 HCCL ranktable`) that look like hardware/HCCL faults.

Usage
-----
    # on the good node
    python3 scripts/verify_rootfs.py manifest /cache/dsv41-rc2-rootfs -o ref.tsv

    # on the copy
    python3 scripts/verify_rootfs.py manifest /cache/dsv41-rc2-rootfs -o copy.tsv
    python3 scripts/verify_rootfs.py verify ref.tsv copy.tsv          # path/type/size/link
    python3 scripts/verify_rootfs.py verify ref.tsv copy.tsv --md5    # + content hashes

Exit code is non-zero when differences are found, so it can gate a deploy.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor


def walk(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        for name in sorted(dirnames):
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                yield path, rel_dir, name
        for name in sorted(filenames):
            yield os.path.join(dirpath, name), rel_dir, name


def md5(path: str) -> str:
    h = hashlib.md5()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError as exc:
        return f"ERR:{exc.errno}"
    return h.hexdigest()


def build_manifest(root: str, with_md5: bool, jobs: int) -> dict[str, str]:
    entries: dict[str, str] = {}
    paths = []
    for path, rel_dir, name in walk(root):
        rel = os.path.join(rel_dir, name) if rel_dir != "." else name
        if os.path.islink(path):
            link = os.readlink(path)
            broken = "" if os.path.exists(path) else "\tBROKEN"
            entries[rel] = f"l\t{len(link)}\t{link}{broken}"
        elif os.path.isdir(path):
            entries[rel] = "d\t0\t"
        elif os.path.isfile(path):
            size = os.path.getsize(path)
            entries[rel] = f"f\t{size}\t"
            if with_md5:
                paths.append((rel, path))
        else:
            entries[rel] = "o\t0\t"
    if with_md5 and paths:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for (rel, _), digest in zip(paths, pool.map(lambda p: md5(p[1]), paths)):
                entries[rel] = entries[rel] + digest
    return entries


def cmd_manifest(args) -> int:
    entries = build_manifest(args.rootfs, args.md5, args.jobs)
    with open(args.out, "w") as fh:
        for rel in sorted(entries):
            fh.write(f"{rel}\t{entries[rel]}\n")
    print(f"wrote {args.out}: {len(entries)} entries"
          f"{' (with md5)' if args.md5 else ''}")
    return 0


def cmd_verify(args) -> int:
    def load(path):
        out = {}
        with open(path) as fh:
            for line in fh:
                rel, rest = line.rstrip("\n").split("\t", 1)
                out[rel] = rest
        return out

    ref, cur = load(args.ref), load(args.cur)
    only_ref = sorted(set(ref) - set(cur))
    only_cur = sorted(set(cur) - set(ref))
    differ = sorted(p for p in set(ref) & set(cur) if ref[p] != cur[p])

    def show(title, items, limit):
        if not items:
            return
        print(f"\n[{title}] {len(items)}")
        for p in items[:limit]:
            print(f"  {p}")
            if p in ref and p in cur and ref[p] != cur[p]:
                print(f"    ref: {ref[p]}")
                print(f"    cur: {cur[p]}")
        if len(items) > limit:
            print(f"  ... {len(items) - limit} more")

    limit = args.limit
    show("missing in copy", only_ref, limit)
    show("extra in copy", only_cur, limit)
    show("different (type/size/link/target/md5)", differ, limit)

    bad = len(only_ref) + len(only_cur) + len(differ)
    print(f"\n{'OK: manifests identical' if bad == 0 else f'MISMATCH: {bad} entries'}")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("manifest", help="write a manifest for one rootfs")
    m.add_argument("rootfs")
    m.add_argument("-o", "--out", required=True)
    m.add_argument("--md5", action="store_true", help="include content hashes (slow)")
    m.add_argument("--jobs", type=int, default=32)
    m.set_defaults(func=cmd_manifest)

    v = sub.add_parser("verify", help="compare two manifests")
    v.add_argument("ref")
    v.add_argument("cur")
    v.add_argument("--md5", action="store_true",
                   help="manifests were produced with --md5 (informational)")
    v.add_argument("--limit", type=int, default=40)
    v.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
